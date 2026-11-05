/* Trigpoint access method: CREATE INDEX (ambuild), the write side of the index pages, and the
 * maintenance entry points that spike S2 does not implement (inserts, vacuum).
 *
 * Build (plan option 2, single-threaded in the leader): one heap scan copies every vector into a
 * malloc'd builder (the detoasted copies live in a short memory context reset per row), the core
 * fits the PCA head, the cells, the codes / norms / router and the calibration
 * (tp_builder_build, polling tp_abort_check between phases), the image is serialised and written
 * as one item per page after the meta page, then WAL-logged with log_newpage_range.
 */
#include "trigpoint.h"

#include "nodes/pg_list.h"

#include <math.h>

#include "access/tableam.h"
#include "executor/tuptable.h"
#include "utils/snapmgr.h"
#include "flat_image.h"
#include "access/generic_xlog.h"
#include "access/xloginsert.h"
#include "catalog/index.h"
#include "optimizer/optimizer.h"
#include "optimizer/plancat.h"
#include "optimizer/paths.h"
#include "access/table.h"
#include "miscadmin.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "storage/freespace.h"
#include "storage/indexfsm.h"
#include "catalog/storage.h"		/* RelationTruncate (E3b) */
#include "utils/memutils.h"
#include "portability/instr_time.h"

#include "extension/vector/vector.h"

#include "tp_graph.h"

typedef struct BuildState
{
	void	   *builder;
	int			D;
	MemoryContext tmpctx;
	int			metric;			/* H11 */
} BuildState;

/* The core's cancellation hook: a read of the interrupt flags, no ereport (this runs in a C++ frame). */
static int
tp_abort_check(void *ctx)
{
	return (QueryCancelPending || ProcDiePending) ? 1 : 0;
}

static void
build_callback(Relation index, ItemPointer tid, Datum *values, bool *isnull, bool tupleIsAlive, void *state)
{
	BuildState *bs = (BuildState *) state;
	MemoryContext old;
	Vector	   *v;
	uint64		t;

	if (isnull[0])
		return;
	old = MemoryContextSwitchTo(bs->tmpctx);
	v = DatumGetVector(values[0]);
	if (v->dim != bs->D)
		ereport(ERROR, (errcode(ERRCODE_DATA_EXCEPTION),
						errmsg("trigpoint: vector has %d dimensions, index column expects %d", v->dim, bs->D)));
	t = ((uint64) ItemPointerGetBlockNumber(tid) << 16) | (uint64) ItemPointerGetOffsetNumber(tid);
	if (bs->metric == TP_METRIC_COSINE)
	{
		/* never in place: an inline datum points into the heap page */
		float	   *xn = (float *) palloc(sizeof(float) * (Size) v->dim);

		memcpy(xn, v->x, sizeof(float) * (Size) v->dim);
		tp_normalize(xn, v->dim);
		if (tp_builder_add(bs->builder, xn, t) != 0)
			ereport(ERROR, (errcode(ERRCODE_OUT_OF_MEMORY), errmsg("trigpoint: out of memory collecting vectors")));
	}
	else if (tp_builder_add(bs->builder, v->x, t) != 0)
		ereport(ERROR, (errcode(ERRCODE_OUT_OF_MEMORY), errmsg("trigpoint: out of memory collecting vectors")));
	MemoryContextSwitchTo(old);
	MemoryContextReset(bs->tmpctx);
}

static int
parse_levels(const char *s, int D, int *lv)
{
	int			L = 0;

	if (s == NULL || *s == '\0')
	{
		/* the experiments' defaults by dimension (run_scale.LEVELS) */
		if (D <= 128) { lv[0] = 16; lv[1] = 32; lv[2] = 64; }
		else { lv[0] = 32; lv[1] = 128; lv[2] = 256; }
		L = 3;
		while (L > 1 && lv[L - 1] > D)
			L--;
		if (lv[L - 1] > D)
			lv[L - 1] = D;
		return L;
	}
	while (*s && L < 4)
	{
		char	   *end;
		long		v = strtol(s, &end, 10);

		if (end == s)
			ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("trigpoint: bad levels option \"%s\"", s)));
		lv[L++] = (int) v;
		s = end;
		while (*s == ',' || *s == ' ')
			s++;
	}
	if (*s || L == 0)
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("trigpoint: levels must be 1 to 4 increasing widths")));
	return L;
}

int
tp_auto_cells(int64 n)
{
	double		c = (double) n / 75.0;
	int			p = 16;

	while (p < 8192 && (double) p * 1.4142 < c)
		p <<= 1;
	while (p > 16 && (int64) p * 8 > n)
		p >>= 1;
	return p;
}

void
tp_init_page(Page page, Size size, uint16 kind, uint32 generation)
{
	TrigpointPageOpaque op;

	PageInit(page, size, sizeof(TrigpointPageOpaqueData));
	op = TrigpointPageGetOpaque(page);
	op->kind = kind;
	op->unused = 0;
	op->generation = generation;
	op->next = InvalidBlockNumber;
}

/*
 * A page to write into (E3): a page the FSM recorded as free (a retired generation's image or log page,
 * see tp_vacuumcleanup) when there is one, else a new page at the end of the fork. The claim is made
 * under the relation extension lock so two backends (a compaction writing its image, an inserter
 * appending a log page) never take the same free page. Returned exclusively locked.
 */
Buffer
tp_new_page(Relation index, ForkNumber fork)
{
	Buffer		buf;

	LockRelationForExtension(index, ExclusiveLock);
	if (fork == MAIN_FORKNUM)
	{
		BlockNumber blk = GetFreeIndexPage(index);

		if (blk != InvalidBlockNumber && blk < RelationGetNumberOfBlocks(index) && blk != TP_METAPAGE_BLKNO)
		{
			RecordUsedIndexPage(index, blk);
			UnlockRelationForExtension(index, ExclusiveLock);
			buf = ReadBuffer(index, blk);
			LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
			return buf;
		}
	}
	buf = ReadBufferExtended(index, fork, P_NEW, RBM_NORMAL, NULL);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	UnlockRelationForExtension(index, ExclusiveLock);
	return buf;
}

/*
 * Write the image chunks as a chain of pages (each page's special `next` names the following one, so the
 * pages need not be contiguous: reused pages come from wherever the FSM has them); *start_out = the first.
 * wal_each: WAL-log every page as it is written (a compaction's scattered pages); the build logs the whole
 * fresh relation as one range instead.
 */
static void
write_image_chain(Relation index, ForkNumber fork, TrigpointMetaPageData *meta, const char *p1, uint64 l1, const char *p2, uint64 l2,
				  bool wal_each, BlockNumber *start_out)
{
	uint64		off = 0,
				total = l1 + l2;
	uint32		i;
	char	   *chunk = (char *) palloc(TP_CHUNK_BYTES);
	Buffer		prev = InvalidBuffer;

	meta->blob_bytes = total;
	meta->nchunks = (uint32) ((total + TP_CHUNK_BYTES - 1) / TP_CHUNK_BYTES);
	*start_out = InvalidBlockNumber;
	for (i = 0; i < meta->nchunks; i++)
	{
		Size		sz = (Size) Min((uint64) TP_CHUNK_BYTES, total - off);
		Buffer		buf;
		Page		page;
		uint64		k;

		CHECK_FOR_INTERRUPTS();
		if (p2 == NULL || off + sz <= l1)
			memcpy(chunk, p1 + off, sz);
		else if (off >= l1)
			memcpy(chunk, p2 + (off - l1), sz);
		else
			for (k = 0; k < sz; k++)
				chunk[k] = (off + k) < l1 ? p1[off + k] : p2[off + k - l1];
		buf = tp_new_page(index, fork);
		if (i == 0)
			*start_out = BufferGetBlockNumber(buf);
		page = BufferGetPage(buf);
		tp_init_page(page, BufferGetPageSize(buf), TP_PAGE_IMAGE, meta->generation);
		if (PageAddItem(page, (Item) chunk, sz, InvalidOffsetNumber, false, false) == InvalidOffsetNumber)
			elog(ERROR, "trigpoint: failed to add image chunk %u to page", i);
		if (BufferIsValid(prev))
		{
			/*
			 * The previous page learns its successor, then goes out. Dirtying a shared buffer and logging it are one
			 * critical section (log_newpage_buffer asserts it; an error between the two would leave a dirty page
			 * without its WAL record).
			 */
			START_CRIT_SECTION();
			TrigpointPageGetOpaque(BufferGetPage(prev))->next = BufferGetBlockNumber(buf);
			MarkBufferDirty(prev);
			if (fork == INIT_FORKNUM || (wal_each && RelationNeedsWAL(index)))
				log_newpage_buffer(prev, true);	/* an init fork is always logged: it is what a crash restores */
			END_CRIT_SECTION();
			UnlockReleaseBuffer(prev);
		}
		prev = buf;
		off += sz;
	}
	if (BufferIsValid(prev))
	{
		START_CRIT_SECTION();
		MarkBufferDirty(prev);
		if (fork == INIT_FORKNUM || (wal_each && RelationNeedsWAL(index)))
			log_newpage_buffer(prev, true);
		END_CRIT_SECTION();
		UnlockReleaseBuffer(prev);
	}
	pfree(chunk);
}

/* The compaction's image: pages from the FSM or the end, WAL-logged one by one. */
void
tp_write_image_pages(Relation index, ForkNumber fork, TrigpointMetaPageData *meta, const char *blob, BlockNumber *start_out)
{
	write_image_chain(index, fork, meta, blob, meta->blob_bytes, NULL, 0, true, start_out);
}

/* Free the pages of a retired generation (its image chain and its log chain) through the FSM: no reader can
 * reach them once the meta page names the next generation (loads hold the meta page's share lock). */
static int
tp_free_generation_pages(Relation index, const TrigpointMetaPageData *old)
{
	int			freed = 0;
	BlockNumber blk,
				nblocks = 0;
	uint32		i;

	blk = old->image_start;
	for (i = 0; i < old->nchunks && BlockNumberIsValid(blk); i++)
	{
		Buffer		buf = tp_read_linked(index, blk, &nblocks);
		BlockNumber next;

		LockBuffer(buf, BUFFER_LOCK_SHARE);
		next = TrigpointPageGetOpaque(BufferGetPage(buf))->next;
		UnlockReleaseBuffer(buf);
		RecordFreeIndexPage(index, blk);
		freed++;
		blk = next;
	}
	blk = old->log_first;
	while (BlockNumberIsValid(blk))
	{
		Buffer		buf = tp_read_linked(index, blk, &nblocks);
		BlockNumber next;

		LockBuffer(buf, BUFFER_LOCK_SHARE);
		next = TrigpointPageGetOpaque(BufferGetPage(buf))->next;
		UnlockReleaseBuffer(buf);
		RecordFreeIndexPage(index, blk);
		freed++;
		blk = next;
	}
	IndexFreeSpaceMapVacuum(index);
	return freed;
}

/* The highest block of a page chain from `start` (limit pages; 0: to the end); InvalidBlockNumber for an empty chain. */
static BlockNumber
tp_chain_max_block(Relation index, BlockNumber start, uint32 limit)
{
	BlockNumber blk = start,
				max = InvalidBlockNumber,
				nblocks = 0;
	uint32		i = 0;

	while (BlockNumberIsValid(blk) && (limit == 0 || i < limit))
	{
		Buffer		buf = tp_read_linked(index, blk, &nblocks);
		BlockNumber next;

		LockBuffer(buf, BUFFER_LOCK_SHARE);
		next = TrigpointPageGetOpaque(BufferGetPage(buf))->next;
		UnlockReleaseBuffer(buf);
		if (!BlockNumberIsValid(max) || blk > max)
			max = blk;
		blk = next;
		if ((++i & 1023) == 0)
			CHECK_FOR_INTERRUPTS();
	}
	return max;
}

/* The blocks of a page chain from `start`, in order (limit pages; 0: to the end), into a palloc'd array; *n_out = count. */
static BlockNumber *
tp_chain_collect(Relation index, BlockNumber start, uint32 limit, uint32 *n_out)
{
	uint32		cap = limit ? limit : 256,
				n = 0;
	BlockNumber *out = (BlockNumber *) palloc(sizeof(BlockNumber) * cap);
	BlockNumber blk = start,
				nblocks = 0;

	while (BlockNumberIsValid(blk) && (limit == 0 || n < limit))
	{
		Buffer		buf = tp_read_linked(index, blk, &nblocks);

		if (n == cap)
		{
			cap *= 2;
			out = (BlockNumber *) repalloc(out, sizeof(BlockNumber) * cap);
		}
		out[n++] = blk;
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		blk = TrigpointPageGetOpaque(BufferGetPage(buf))->next;
		UnlockReleaseBuffer(buf);
		if ((n & 1023) == 0)
			CHECK_FOR_INTERRUPTS();
	}
	*n_out = n;
	return out;
}

/*
 * E3b (2026-11-02): give a retired generation's space back to the operating system. The compaction writes
 * the next generation's image before the old one can be freed, so the new chain mostly lands past the old
 * image, and once the old image is freed its pages sit below the live ones: the FSM reuses them for the
 * generation after, but the file never shrinks (1M x 1,536: 856 MB at build, 1,633 MB after the first
 * compaction and still after the second). Run after the flip and the free:
 *   1. The live pages are the meta page, the image chain and the log chain; T = their count is the size the
 *      file can have. Under the meta page's exclusive lock (appenders claim their pages under it, so none can
 *      take ours meanwhile) the log chain is walked, every free block below T is listed as a target and marked
 *      used in the FSM.
 *   2. Every image page at or above T is copied into the next target, last page first, so each copy is
 *      written with its final `next` and only the predecessor of each moved run (an unmoved page, or the meta
 *      page's image_start) is updated in place. The chain is intact at every instant: a load that read a
 *      predecessor before its update walks the old pages, which are not freed yet.
 *   3. The meta page's exclusive lock is taken once more, which waits for such loads (they hold its share
 *      lock), then the old pages go to the FSM.
 *   4. The relation is truncated after its highest live page under a conditional AccessExclusiveLock, as the
 *      heap's VACUUM does: a backend that holds the index open makes this VACUUM skip the truncation, the
 *      next one gets it. Returns the pages truncated; *moved_out = image pages moved.
 */
static BlockNumber
tp_reclaim_space(Relation index, TrigpointMetaPageData *meta, uint32 *moved_out)
{
	BlockNumber nblocks,
				cutoff,
				truncated = 0,
				T;
	BlockNumber *chain,
			   *logs,
			   *targets,
			   *oldblk;
	uint32		nchain,
				nlog,
				ntargets = 0,
				nmove = 0,
				i,
				k;
	uint8	   *used;
	Buffer		buf;
	TrigpointMetaPageData cur;

	*moved_out = 0;
	chain = tp_chain_collect(index, meta->image_start, meta->nchunks, &nchain);
	if (nchain != meta->nchunks)
		ereport(ERROR, (errcode(ERRCODE_INDEX_CORRUPTED), errmsg("trigpoint: image chain of \"%s\" has %u pages, meta says %u", RelationGetRelationName(index), nchain, meta->nchunks), errhint("Run trigpoint_check(<index>, true); REINDEX rebuilds the index.")));
	/* 1. the live set and the targets, with appenders held off */
	buf = ReadBuffer(index, TP_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	memcpy(&cur, TrigpointPageGetMeta(BufferGetPage(buf)), sizeof(cur));
	if (cur.generation != meta->generation || cur.image_start != meta->image_start)
		elog(ERROR, "trigpoint: generation changed under the reclaim of \"%s\"", RelationGetRelationName(index));
	nblocks = RelationGetNumberOfBlocks(index);
	logs = tp_chain_collect(index, cur.log_first, 0, &nlog);
	used = (uint8 *) palloc0(nblocks);
	used[TP_METAPAGE_BLKNO] = 1;
	for (i = 0; i < nchain; i++)
		if (chain[i] < nblocks)
			used[chain[i]] = 1;
	for (i = 0; i < nlog; i++)
		if (logs[i] < nblocks)
			used[logs[i]] = 1;
	T = 1 + nchain + nlog;
	if (T > nblocks)
		T = nblocks;
	for (i = 0; i < nchain; i++)
		if (chain[i] >= T)
			nmove++;
	targets = (BlockNumber *) palloc(sizeof(BlockNumber) * (nmove ? nmove : 1));
	for (i = TP_METAPAGE_BLKNO + 1; i < T && ntargets < nmove; i++)
		if (!used[i])
		{
			targets[ntargets++] = i;
			RecordUsedIndexPage(index, i);
		}
	UnlockReleaseBuffer(buf);
	pfree(used);
	pfree(logs);
	if (ntargets < nmove)
		nmove = ntargets;		/* cannot happen by the count (free pages below T = live pages at or above T); keep the rest in place */
	/* 2. the moves, last page first; targets in chain order */
	oldblk = (BlockNumber *) palloc(sizeof(BlockNumber) * (nmove ? nmove : 1));
	k = nmove;
	for (i = nchain; i-- > 0 && k > 0;)
	{
		Buffer		sbuf,
					dbuf;
		Page		spage,
					dpage;
		BlockNumber dst;

		if (chain[i] < T)
			continue;
		dst = targets[--k];
		oldblk[k] = chain[i];
		sbuf = ReadBuffer(index, chain[i]);
		LockBuffer(sbuf, BUFFER_LOCK_SHARE);
		dbuf = ReadBuffer(index, dst);
		LockBuffer(dbuf, BUFFER_LOCK_EXCLUSIVE);
		spage = BufferGetPage(sbuf);
		dpage = BufferGetPage(dbuf);
		START_CRIT_SECTION();
		memcpy(dpage, spage, BLCKSZ);
		TrigpointPageGetOpaque(dpage)->next = (i + 1 < nchain) ? chain[i + 1] : InvalidBlockNumber;	/* chain[i+1] is final: moved pages were rewritten first */
		MarkBufferDirty(dbuf);
		if (RelationNeedsWAL(index))
			log_newpage_buffer(dbuf, true);
		END_CRIT_SECTION();
		UnlockReleaseBuffer(dbuf);
		UnlockReleaseBuffer(sbuf);
		chain[i] = dst;
		/* the predecessor learns the new location now only if it stays: a moved predecessor is written with it */
		if (i == 0)
		{
			GenericXLogState *state;
			Page		page;

			buf = ReadBuffer(index, TP_METAPAGE_BLKNO);
			LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
			state = GenericXLogStart(index);
			page = GenericXLogRegisterBuffer(state, buf, 0);
			TrigpointPageGetMeta(page)->image_start = dst;
			GenericXLogFinish(state);
			UnlockReleaseBuffer(buf);
			meta->image_start = dst;
		}
		else if (chain[i - 1] < T)
		{
			GenericXLogState *state;
			Page		page;
			Buffer		pbuf = ReadBuffer(index, chain[i - 1]);

			LockBuffer(pbuf, BUFFER_LOCK_EXCLUSIVE);
			state = GenericXLogStart(index);
			page = GenericXLogRegisterBuffer(state, pbuf, 0);
			TrigpointPageGetOpaque(page)->next = dst;
			GenericXLogFinish(state);
			UnlockReleaseBuffer(pbuf);
		}
		if ((k & 1023) == 0)
			CHECK_FOR_INTERRUPTS();
	}
	*moved_out = nmove;
	if (nmove > 0)
	{
		/* 3. wait for loads that may still walk the old pages, then free them */
		buf = ReadBuffer(index, TP_METAPAGE_BLKNO);
		LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
		UnlockReleaseBuffer(buf);
		for (k = 0; k < nmove; k++)
			RecordFreeIndexPage(index, oldblk[k]);
		IndexFreeSpaceMapVacuum(index);
	}
	pfree(oldblk);
	pfree(targets);
	pfree(chain);
	/* 4. the truncation */
	if (!ConditionalLockRelation(index, AccessExclusiveLock))
		return 0;
	nblocks = RelationGetNumberOfBlocks(index);
	buf = ReadBuffer(index, TP_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	memcpy(&cur, TrigpointPageGetMeta(BufferGetPage(buf)), sizeof(cur));
	UnlockReleaseBuffer(buf);
	{
		BlockNumber imax = tp_chain_max_block(index, cur.image_start, cur.nchunks),
					lmax = tp_chain_max_block(index, cur.log_first, 0);

		cutoff = TP_METAPAGE_BLKNO + 1;
		if (BlockNumberIsValid(imax) && imax + 1 > cutoff)
			cutoff = imax + 1;
		if (BlockNumberIsValid(lmax) && lmax + 1 > cutoff)
			cutoff = lmax + 1;
	}
	if (cutoff < nblocks)
	{
		RelationTruncate(index, cutoff);
		truncated = nblocks - cutoff;
	}
	UnlockRelation(index, AccessExclusiveLock);
	return truncated;
}

/* As tp_write_image_pages for a blob made of two regions (p1 of l1 bytes, then p2 of l2): the parallel build's
 * TIDs and image live in different parts of its shared area. */
static void
write_image_pages2(Relation index, ForkNumber fork, TrigpointMetaPageData *meta, const char *p1, uint64 l1, const char *p2, uint64 l2, BlockNumber *start_out)
{
	write_image_chain(index, fork, meta, p1, l1, p2, l2, false, start_out);
}

int
tp_cmp_float(const void *a, const void *b)
{
	float		x = *(const float *) a,
				y = *(const float *) b;

	return x < y ? -1 : (x > y ? 1 : 0);
}

/* CREATE INDEX: block 0 = meta, then generation 0's image; the log is empty. */
static void
write_index(Relation index, ForkNumber fork, TrigpointMetaPageData *meta, const char *blob)
{
	Buffer		buf;
	Page		page;

	meta->generation = 0;
	meta->log_first = meta->log_last = InvalidBlockNumber;
	meta->n_image = meta->n;
	buf = ReadBufferExtended(index, fork, P_NEW, RBM_NORMAL, NULL);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	if (BufferGetBlockNumber(buf) != TP_METAPAGE_BLKNO)
		elog(ERROR, "trigpoint: meta page is not block 0");
	page = BufferGetPage(buf);
	PageInit(page, BufferGetPageSize(buf), 0);
	if (blob)
	{
		UnlockReleaseBuffer(buf);		/* write the image first so the meta page can name its start */
		write_image_chain(index, fork, meta, blob, meta->blob_bytes, NULL, 0, false, &meta->image_start);
		buf = ReadBufferExtended(index, fork, TP_METAPAGE_BLKNO, RBM_NORMAL, NULL);
		LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
		page = BufferGetPage(buf);
		PageInit(page, BufferGetPageSize(buf), 0);
	}
	else
	{
		meta->nchunks = 0;
		meta->image_start = InvalidBlockNumber;
	}
	memcpy(TrigpointPageGetMeta(page), meta, sizeof(TrigpointMetaPageData));
	((PageHeader) page)->pd_lower = ((char *) TrigpointPageGetMeta(page) + sizeof(TrigpointMetaPageData)) - (char *) page;
	START_CRIT_SECTION();
	MarkBufferDirty(buf);
	if (fork == INIT_FORKNUM)
		log_newpage_buffer(buf, true);
	END_CRIT_SECTION();
	UnlockReleaseBuffer(buf);
	if (fork == MAIN_FORKNUM && RelationNeedsWAL(index))
		log_newpage_range(index, MAIN_FORKNUM, 0, RelationGetNumberOfBlocks(index), true);
}

/* The parallel build's write: meta page, then the image from two regions (see write_index). */
void
tp_write_index2(Relation index, ForkNumber fork, TrigpointMetaPageData *meta, const char *p1, uint64 l1, const char *p2, uint64 l2, BlockNumber *start_out)
{
	Buffer		buf;
	Page		page;

	meta->generation = 0;
	meta->log_first = meta->log_last = InvalidBlockNumber;
	meta->n_image = meta->n;
	buf = ReadBufferExtended(index, fork, P_NEW, RBM_NORMAL, NULL);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	if (BufferGetBlockNumber(buf) != TP_METAPAGE_BLKNO)
		elog(ERROR, "trigpoint: meta page is not block 0");
	UnlockReleaseBuffer(buf);
	write_image_pages2(index, fork, meta, p1, l1, p2, l2, start_out);
	meta->image_start = *start_out;
	buf = ReadBufferExtended(index, fork, TP_METAPAGE_BLKNO, RBM_NORMAL, NULL);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	page = BufferGetPage(buf);
	PageInit(page, BufferGetPageSize(buf), 0);
	memcpy(TrigpointPageGetMeta(page), meta, sizeof(TrigpointMetaPageData));
	((PageHeader) page)->pd_lower = ((char *) TrigpointPageGetMeta(page) + sizeof(TrigpointMetaPageData)) - (char *) page;
	MarkBufferDirty(buf);
	UnlockReleaseBuffer(buf);
	if (RelationNeedsWAL(index))
		log_newpage_range(index, fork, 0, RelationGetNumberOfBlocks(index), true);
}

/*
 * Workers for the build, pgvector's rule (HnswParallelBuildWorkers) with one difference: the planner's
 * plan_create_index_workers sizes the table by its MAIN fork alone (min_parallel_table_scan_size, 8 MB),
 * and a TOASTed vector column leaves that fork tiny (75k x 1536: 484 pages beside 75,924 TOAST pages),
 * so it answers 0 for exactly the tables this index is for. We count the TOAST relation's pages too.
 * Then, as pgvector: the table's parallel_workers reloption if set, else max_parallel_maintenance_workers.
 */
static int
tp_build_workers(Relation heap, Relation index)
{
	int			w = plan_create_index_workers(RelationGetRelid(heap), RelationGetRelid(index));
	int			ropt;

	if (w == 0)
	{
		BlockNumber pages = RelationGetNumberOfBlocks(heap);

		if (OidIsValid(heap->rd_rel->reltoastrelid))
		{
			Relation	toast = table_open(heap->rd_rel->reltoastrelid, AccessShareLock);

			pages += RelationGetNumberOfBlocks(toast);
			table_close(toast, AccessShareLock);
		}
		if (max_parallel_maintenance_workers == 0 || pages < (BlockNumber) min_parallel_table_scan_size)
			return 0;
	}
	ropt = RelationGetParallelWorkers(heap, -1);
	if (ropt != -1)
		return Min(ropt, max_parallel_maintenance_workers);
	return max_parallel_maintenance_workers;
}

IndexBuildResult *
tp_build(Relation heap, Relation index, IndexInfo *indexInfo)
{
	TrigpointOptions *opts = (TrigpointOptions *) index->rd_options;
	IndexBuildResult *result;
	BuildState	bs;
	TrigpointMetaPageData meta;
	int			D = TupleDescAttr(index->rd_att, 0)->atttypmod;
	int			prm[15];
	double		st[6] = {0, 0, 0, 0, 0, 0};
	void	   *ix = NULL;
	size_t		core_len = 0;
	char	   *blob = NULL;
	const uint64 *tids;
	int64		n;
	int			l;
	instr_time	t0, t1, t2, t3;

	if (D < 1)
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("trigpoint: the indexed column must have fixed dimensions, e.g. vector(1536)")));
	if (indexInfo->ii_NumIndexAttrs != 1)
		ereport(ERROR, (errcode(ERRCODE_FEATURE_NOT_SUPPORTED), errmsg("trigpoint: one indexed column only")));

	memset(&meta, 0, sizeof(meta));
	meta.magic = TP_MAGIC;
	meta.version = TP_META_VERSION;
	meta.D = D;
	meta.L = parse_levels(opts ? GET_STRING_RELOPTION(opts, levels_offset) : NULL, D, meta.lv);
	for (l = 0; l < meta.L; l++)
		if (meta.lv[l] < 1 || meta.lv[l] > D || (l && meta.lv[l] <= meta.lv[l - 1]))
			ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							errmsg("trigpoint: levels must be increasing and at most the dimension (%d)", D)));
	meta.M = opts ? opts->m : 32;
	meta.efC = opts ? opts->ef_construction : 200;
	meta.metric = tp_index_metric(index);

	/* E2: the worker build is the default; the core's threads only on request (measurement shortcut) */
	if (!(tp_allow_build_threads && opts && opts->threads > 1))
	{
		int			request = tp_build_workers(heap, index),
					nparticipants = 1;
		double		ph[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
		int64		nrows;
		instr_time	tb0,
					tb1;

		INSTR_TIME_SET_CURRENT(tb0);
		nrows = tp_parallel_build(heap, index, indexInfo, &meta, opts, request, &nparticipants, ph);
		INSTR_TIME_SET_CURRENT(tb1);
		if (nrows == 0)
		{
			/* H14: an empty table gets the trivial model, so inserts work (TRUNCATE, create-then-load) */
			tp_write_trivial(index, MAIN_FORKNUM, &meta);
			result = (IndexBuildResult *) palloc0(sizeof(IndexBuildResult));
			return result;
		}
		/* phase timings into the meta page: scan (incl. the counting scan + DSM setup), fit, head, kmeans, codes + graph, calib, write */
		meta.t_scan = ph[0] + ph[7];
		meta.t_fit = ph[1];
		meta.t_head = ph[2];
		meta.t_kmeans = ph[3];
		meta.t_index = ph[4] + ph[5];
		meta.t_calib = ph[6];
		meta.t_write = ph[8];
		meta.t_codes = ph[4];
		meta.t_graph = ph[5];
		meta.participants = nparticipants;
		{
			Buffer		buf = ReadBuffer(index, TP_METAPAGE_BLKNO);

			LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
			memcpy(TrigpointPageGetMeta(BufferGetPage(buf)), &meta, sizeof(meta));
			START_CRIT_SECTION();
			MarkBufferDirty(buf);
			if (RelationNeedsWAL(index))
				log_newpage_buffer(buf, true);
			END_CRIT_SECTION();
			UnlockReleaseBuffer(buf);
		}
		ereport(DEBUG1, (errmsg("trigpoint: built %d rows, %d cells, %u pages with %d participants in %.1fs; scan %.1fs fit %.1fs head %.1fs kmeans %.1fs codes %.1fs graph %.1fs calib %.1fs write %.1fs",
								meta.n, meta.C, meta.nchunks + 1, nparticipants, INSTR_TIME_GET_DOUBLE(tb1) - INSTR_TIME_GET_DOUBLE(tb0),
								meta.t_scan, meta.t_fit, meta.t_head, meta.t_kmeans, ph[4], ph[5], meta.t_calib, meta.t_write)));
		result = (IndexBuildResult *) palloc(sizeof(IndexBuildResult));
		result->heap_tuples = (double) nrows;
		result->index_tuples = (double) nrows;
		return result;
	}

	bs.D = D;
	bs.metric = meta.metric;
	bs.tmpctx = AllocSetContextCreate(CurrentMemoryContext, "trigpoint build row", ALLOCSET_DEFAULT_SIZES);
	bs.builder = tp_builder_new(D, (int64) heap->rd_rel->reltuples);
	if (bs.builder == NULL)
		ereport(ERROR, (errcode(ERRCODE_OUT_OF_MEMORY), errmsg("trigpoint: %s", tp_last_error())));

	PG_TRY();
	{
		INSTR_TIME_SET_CURRENT(t0);
		n = (int64) table_index_build_scan(heap, index, indexInfo, true, true, build_callback, &bs, NULL);
		INSTR_TIME_SET_CURRENT(t1);
		n = tp_builder_count(bs.builder);	/* non-null rows */
		if (n < 1)
			ereport(ERROR, (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
							errmsg("trigpoint: cannot build an index over an empty table in spike S2 (no inserts yet)")));
		if (n > INT32_MAX)
			ereport(ERROR, (errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED), errmsg("trigpoint: too many rows")));
		meta.n = (int32) n;
		meta.C = (opts && opts->cells > 0) ? opts->cells : tp_auto_cells(n);
		if (meta.C > n)
			meta.C = (int) n;

		prm[0] = meta.C;
		prm[1] = meta.L;
		for (l = 0; l < 4; l++)
			prm[2 + l] = l < meta.L ? meta.lv[l] : 0;
		prm[6] = meta.M;
		prm[7] = meta.efC;
		prm[8] = opts ? opts->sample : 200000;
		prm[9] = opts ? opts->kmeans_iters : 15;
		prm[10] = 100000;		/* k-means training rows */
		prm[11] = opts ? opts->calibration_queries : 2000;
		prm[12] = 30;			/* subspace iterations of the PCA fit */
		prm[13] = 0;			/* seed */
		prm[14] = opts ? opts->threads : 1;	/* build threads inside the core */

		ix = tp_builder_build(bs.builder, prm, st, tp_abort_check, NULL);
		if (ix == NULL)
		{
			CHECK_FOR_INTERRUPTS();	/* a cancelled build reports here */
			ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("trigpoint: build failed: %s", tp_last_error())));
		}
		INSTR_TIME_SET_CURRENT(t2);
		{
			long long	flat = tp_index_flatten(ix, NULL, 0);	/* v3 pages: TIDs then the flat image */

			if (flat < 0)
				ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("trigpoint: %s", tp_last_error())));
			core_len = (size_t) flat;
		}
		tp_index_ch(ix, 0.01, meta.ch_default);
		meta.var_captured = (float) st[5];
		meta.core_bytes = core_len;
		meta.blob_bytes = (uint64) n * sizeof(uint64) + core_len;

		/* image = tids then the flat core image; assembled in a malloc'd buffer (may exceed MaxAllocSize) */
		blob = (char *) malloc(meta.blob_bytes);
		if (blob == NULL)
			ereport(ERROR, (errcode(ERRCODE_OUT_OF_MEMORY), errmsg("trigpoint: out of memory for the index image")));
		tids = tp_builder_tids(bs.builder);
		memcpy(blob, tids, (size_t) n * sizeof(uint64));
		if (tp_index_flatten(ix, blob + (size_t) n * sizeof(uint64), core_len) != (long long) core_len)
			ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("trigpoint: %s", tp_last_error())));
		tp_index_free(ix);
		ix = NULL;
		tp_builder_free(bs.builder);
		bs.builder = NULL;

		meta.t_scan = INSTR_TIME_GET_DOUBLE(t1) - INSTR_TIME_GET_DOUBLE(t0);
		meta.t_fit = st[0];
		meta.t_head = st[1];
		meta.t_kmeans = st[2];
		meta.t_index = st[3];
		meta.t_calib = st[4];
		write_index(index, MAIN_FORKNUM, &meta, blob);
		INSTR_TIME_SET_CURRENT(t3);
		meta.t_write = INSTR_TIME_GET_DOUBLE(t3) - INSTR_TIME_GET_DOUBLE(t2);
		{
			/* rewrite the meta page with the final timing */
			Buffer		buf = ReadBuffer(index, TP_METAPAGE_BLKNO);

			LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
			memcpy(TrigpointPageGetMeta(BufferGetPage(buf)), &meta, sizeof(meta));
			START_CRIT_SECTION();
			MarkBufferDirty(buf);
			if (RelationNeedsWAL(index))
				log_newpage_buffer(buf, true);
			END_CRIT_SECTION();
			UnlockReleaseBuffer(buf);
		}
		free(blob);
		blob = NULL;
	}
	PG_CATCH();
	{
		if (blob)
			free(blob);
		if (ix)
			tp_index_free(ix);
		if (bs.builder)
			tp_builder_free(bs.builder);
		PG_RE_THROW();
	}
	PG_END_TRY();
	MemoryContextDelete(bs.tmpctx);

	ereport(DEBUG1, (errmsg("trigpoint: built %d rows, %d cells, %u pages; scan %.1fs fit %.1fs head %.1fs kmeans %.1fs index %.1fs calib %.1fs write %.1fs",
							meta.n, meta.C, meta.nchunks + 1, meta.t_scan, meta.t_fit, meta.t_head, meta.t_kmeans, meta.t_index,
							meta.t_calib, meta.t_write)));

	result = (IndexBuildResult *) palloc(sizeof(IndexBuildResult));
	result->heap_tuples = (double) n;
	result->index_tuples = (double) n;
	return result;
}

/*
 * H14: the image of an index over no rows -- the core's trivial model (identity projection on the first d3
 * coordinates, one cell, exact bound only) with n = 0, so that inserts go through the log as usual. Used by
 * CREATE INDEX on an empty table and by the UNLOGGED init fork. meta carries D, L, lv, M and efC.
 */
void
tp_write_trivial(Relation index, ForkNumber fork, TrigpointMetaPageData *meta)
{
	void	   *ix = tp_index_build_trivial(meta->D, meta->L, meta->lv, meta->M, meta->efC);
	long long	flat;
	char	   *blob = NULL;

	if (ix == NULL)
		ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("trigpoint: %s", tp_last_error())));
	PG_TRY();
	{
		flat = tp_index_flatten(ix, NULL, 0);
		if (flat < 0)
			ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("trigpoint: %s", tp_last_error())));
		blob = (char *) malloc((size_t) flat);
		if (blob == NULL)
			ereport(ERROR, (errcode(ERRCODE_OUT_OF_MEMORY), errmsg("trigpoint: out of memory for the index image")));
		if (tp_index_flatten(ix, blob, (size_t) flat) != flat)
			ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("trigpoint: %s", tp_last_error())));
		meta->n = 0;
		meta->C = 1;
		meta->core_bytes = (uint64) flat;
		meta->blob_bytes = (uint64) flat;
		tp_index_ch(ix, 0.01, meta->ch_default);
		meta->var_captured = 0;
		meta->participants = 1;
		write_index(index, fork, meta, blob);
	}
	PG_CATCH();
	{
		if (blob)
			free(blob);
		tp_index_free(ix);
		PG_RE_THROW();
	}
	PG_END_TRY();
	free(blob);
	tp_index_free(ix);
}

/* UNLOGGED tables: an init fork holding the trivial model (H14), so the index takes inserts after a crash. */
void
tp_buildempty(Relation index)
{
	TrigpointMetaPageData meta;
	TrigpointOptions *opts = (TrigpointOptions *) index->rd_options;
	int			l;

	memset(&meta, 0, sizeof(meta));
	meta.magic = TP_MAGIC;
	meta.version = TP_META_VERSION;
	meta.D = TupleDescAttr(index->rd_att, 0)->atttypmod;
	if (meta.D < 1)
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("trigpoint: the indexed column must have fixed dimensions, e.g. vector(1536)")));
	meta.L = parse_levels(opts ? GET_STRING_RELOPTION(opts, levels_offset) : NULL, meta.D, meta.lv);
	for (l = 0; l < meta.L; l++)
		if (meta.lv[l] < 1 || meta.lv[l] > meta.D || (l && meta.lv[l] <= meta.lv[l - 1]))
			ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							errmsg("trigpoint: levels must be increasing and at most the dimension (%d)", meta.D)));
	meta.M = opts ? opts->m : 32;
	meta.efC = opts ? opts->ef_construction : 200;
	meta.metric = tp_index_metric(index);
	tp_write_trivial(index, INIT_FORKNUM, &meta);
}

/*
 * aminsert: encode the vector with this backend's copy of the index (loaded and synced if needed),
 * append an INSERT record to the log (WAL-logged through generic xlog) and apply it to the copy.
 * The id is implicit: n_image + ordinal in the log, so every backend assigns the same one.
 */
bool
tp_insert(Relation index, Datum *values, bool *isnull, ItemPointer heap_tid, Relation heap,
		  IndexUniqueCheck checkUnique, bool indexUnchanged, IndexInfo *indexInfo)
{
	TrigpointCached *c;
	Vector	   *v;
	TrigpointLogRecord *rec;
	Size		size;
	int			d3,
				cell;

	if (isnull[0])
		return false;
	c = tp_cache_get(index);
	PG_TRY();
	{
		if (c->ix == NULL)
			ereport(ERROR, (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
							errmsg("trigpoint: cannot insert into an empty index written by trigpoint 0.2"),
							errhint("REINDEX the index.")));
		v = DatumGetVector(values[0]);
		if (v->dim != c->D)
			ereport(ERROR, (errcode(ERRCODE_DATA_EXCEPTION), errmsg("trigpoint: vector has %d dimensions, index expects %d", v->dim, c->D)));
		d3 = c->meta.lv[c->L - 1];
		size = TP_INSERT_RECORD_SIZE(d3);
		rec = (TrigpointLogRecord *) palloc0(size);
		rec->kind = TP_REC_INSERT;
		rec->tid = ((uint64) ItemPointerGetBlockNumber(heap_tid) << 16) | (uint64) ItemPointerGetOffsetNumber(heap_tid);
		if (c->meta.metric == TP_METRIC_COSINE)
		{
			float	   *xn = (float *) palloc(sizeof(float) * (Size) c->D);

			memcpy(xn, v->x, sizeof(float) * (Size) c->D);
			tp_normalize(xn, c->D);
			cell = tp_index_encode(c->ix, xn, rec->codes, rec->norms, rec->norms + 4);
			pfree(xn);
		}
		else
			cell = tp_index_encode(c->ix, v->x, rec->codes, rec->norms, rec->norms + 4);
		if (cell < 0)
			ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("trigpoint: %s", tp_last_error())));
		rec->cell = (uint16) cell;
		tp_log_append(index, c, rec, size);	/* syncs, appends, applies, under the log tail's lock */
		pfree(rec);
	}
	PG_CATCH();
	{
		tp_cache_unpin(c);
		PG_RE_THROW();
	}
	PG_END_TRY();
	tp_cache_unpin(c);
	return false;
}

/*
 * ambulkdelete: ask the callback about every live TID this copy knows (the image's and the log's),
 * tombstone the dead ones in the log (DELETE records) and in the copy. Dead ids are never fetched
 * again; their space is given back by the compaction in amvacuumcleanup.
 */
IndexBulkDeleteResult *
tp_bulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats, IndexBulkDeleteCallback callback, void *callback_state)
{
	Relation	index = info->index;
	TrigpointCached *c;
	int			i,
				removed = 0;

	if (stats == NULL)
		stats = (IndexBulkDeleteResult *) palloc0(sizeof(IndexBulkDeleteResult));
	c = tp_cache_get(index);
	PG_TRY();
	{
		TrigpointLogRecord rec;

		memset(&rec, 0, sizeof(rec));
		rec.kind = TP_REC_DELETE;
		for (i = 0; i < c->n; i++)
		{
			ItemPointerData tid;

			if (!tp_tid_of(c, i, &tid) || tp_index_is_deleted(c->ix, i))
				continue;			/* already tombstoned (the shared TID table is read-only: ask the core) */
			if (callback(&tid, callback_state))
			{
				rec.id = i;
				tp_log_append(index, c, &rec, TP_DELETE_RECORD_SIZE);
				removed++;
			}
			if ((i & 4095) == 0)
				CHECK_FOR_INTERRUPTS();
		}
	}
	PG_CATCH();
	{
		tp_cache_unpin(c);
		PG_RE_THROW();
	}
	PG_END_TRY();
	tp_cache_unpin(c);
	stats->tuples_removed += removed;
	stats->num_index_tuples = c->n;
	ereport(DEBUG1, (errmsg("trigpoint: vacuum of \"%s\" tombstoned %d rows", RelationGetRelationName(index), removed)));
	return stats;
}

/*
 * Lost-write fix (2026-10-26, after E1): records appended to the log between the compaction's sync and
 * its flip would be dropped by a flip that resets log_first / log_last wholesale. Under the meta page's
 * exclusive lock (appenders wait on it), copy every record after the position this compaction consumed
 * ((c->log_blk, c->log_off), exclusive) into fresh log pages tagged with the new generation, and make
 * them the new generation's log. The records' ids continue from the image's n (appenders assigned them
 * sequentially after the ones we consumed), so a backend loading the new generation replays them as
 * its first log records. Returns the number of records carried.
 */
static int
tp_carry_log(Relation index, TrigpointCached *c, TrigpointMetaPageData *newmeta, const TrigpointMetaPageData *cur,
			 const int32 *newpos, int n_old)
{
	BlockNumber blk,
				nblocks = 0;
	OffsetNumber from;
	List	   *recs = NIL;
	ListCell   *lc;
	Buffer		prev = InvalidBuffer;
	int			carried = 0,
				dropped = 0,
				next_id = newmeta->n;

	if (!BlockNumberIsValid(cur->log_first))
		return 0;
	blk = BlockNumberIsValid(c->log_blk) ? c->log_blk : cur->log_first;
	from = (blk == c->log_blk) ? c->log_off : InvalidOffsetNumber;
	/* collect copies of the unconsumed records (page by page under share locks) */
	for (;;)
	{
		Buffer		buf = tp_read_linked(index, blk, &nblocks);
		Page		page;
		OffsetNumber off,
					max;
		BlockNumber next;

		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		max = PageGetMaxOffsetNumber(page);
		for (off = (from == InvalidOffsetNumber) ? FirstOffsetNumber : from + 1; off <= max; off++)
		{
			ItemId		iid = PageGetItemId(page, off);
			Size		sz;
			TrigpointLogRecord *copy;

			if (!ItemIdIsUsed(iid))
				continue;
			sz = ItemIdGetLength(iid);
			copy = (TrigpointLogRecord *) palloc(sz);
			memcpy(copy, PageGetItem(page, iid), sz);
			/* H13: the new generation renumbers ids (id = position among the live rows). An INSERT appended after our
			 * sync gets the next new id; a tombstone names either such an insert (old id >= n_old: the same offset past
			 * the new count) or a row of the image (mapped through its position; dropped when that row is dead already). */
			if (copy->kind == TP_REC_INSERT)
				copy->id = next_id++;
			else
			{
				int32		old = copy->id,
							nid = -1;

				if (old >= n_old)
					nid = newmeta->n + (old - n_old);
				else
				{
					int			pos = tp_index_id_pos(c->ix, old);

					nid = (pos >= 0) ? newpos[pos] : -1;
				}
				if (nid < 0)
				{
					pfree(copy);
					dropped++;
					continue;
				}
				copy->id = nid;
			}
			recs = lappend(recs, copy);
		}
		next = TrigpointPageGetOpaque(page)->next;
		UnlockReleaseBuffer(buf);
		if (!BlockNumberIsValid(next))
			break;
		blk = next;
		from = InvalidOffsetNumber;
	}
	/* write them into new pages of the new generation, chained; each new page is one generic-xlog record
	 * (with the previous page's next pointer), written before the meta page that will reference them */
	lc = list_head(recs);
	while (lc != NULL)
	{
		GenericXLogState *state = GenericXLogStart(index);
		Buffer		nbuf;
		Page		npage;

		nbuf = tp_new_page(index, MAIN_FORKNUM);
		npage = GenericXLogRegisterBuffer(state, nbuf, GENERIC_XLOG_FULL_IMAGE);
		tp_init_page(npage, BufferGetPageSize(nbuf), TP_PAGE_LOG, newmeta->generation);
		if (BufferIsValid(prev))
		{
			Page		ppage = GenericXLogRegisterBuffer(state, prev, 0);

			TrigpointPageGetOpaque(ppage)->next = BufferGetBlockNumber(nbuf);
		}
		else
			newmeta->log_first = BufferGetBlockNumber(nbuf);
		newmeta->log_last = BufferGetBlockNumber(nbuf);
		while (lc != NULL)
		{
			TrigpointLogRecord *rec = (TrigpointLogRecord *) lfirst(lc);
			Size		sz = (rec->kind == TP_REC_INSERT) ? TP_INSERT_RECORD_SIZE(newmeta->lv[newmeta->L - 1]) : TP_DELETE_RECORD_SIZE;

			if (PageGetFreeSpace(npage) < sz + sizeof(ItemIdData))
				break;
			if (PageAddItem(npage, (Item) rec, sz, InvalidOffsetNumber, false, false) == InvalidOffsetNumber)
				elog(ERROR, "trigpoint: failed to carry a log record into the new generation");
			if (rec->kind == TP_REC_INSERT)
				newmeta->log_inserts++;
			else
				newmeta->log_deletes++;
			carried++;
			lc = lnext(recs, lc);
		}
		GenericXLogFinish(state);
		if (BufferIsValid(prev))
			UnlockReleaseBuffer(prev);
		prev = nbuf;
	}
	if (BufferIsValid(prev))
		UnlockReleaseBuffer(prev);
	list_free_deep(recs);
	if (dropped)
		ereport(DEBUG1, (errmsg("trigpoint: %d tombstones of rows already dead dropped from the carried log", dropped)));
	return carried;
}

/*
 * H16: re-calibration of a compacted image over the rows it now holds. The build fits the calibration
 * tables (critical cosines of (query, true neighbour) pairs per level) once; when the data drifts the
 * calibrated bound over-prunes true neighbours (3-minute soak with drifting inserts: recall 0.91 with every
 * cell probed, 1.00 at exact = on and after REINDEX). A compaction has every live row's TID, so it samples
 * `m` of them, fetches their vectors from the heap, finds each one's exact top-CAL_K through the new image's
 * own certificate scan (exact bound, every cell, exact distances from fetched vectors) and rewrites the
 * tables and the default c_h in the image header. Rows that vanished meanwhile are skipped.
 */
#define TP_CAL_K 10
#define TP_CAL_SLACK (1.0f + 1e-5f)

/* the row's vector as a private copy (an inline datum would point into the heap page) */
static bool
fetch_vector(Relation heap, TupleTableSlot *slot, AttrNumber attno, uint64 t, Vector **out)
{
	ItemPointerData tid;
	bool		isnull;
	Datum		d;

	if (t == 0)
		return false;
	ItemPointerSet(&tid, (BlockNumber) (t >> 16), (OffsetNumber) (t & 0xFFFF));
	if (ItemPointerGetBlockNumber(&tid) >= RelationGetNumberOfBlocks(heap))
		return false;
	if (!table_tuple_fetch_row_version(heap, &tid, SnapshotAny, slot))
		return false;
	d = slot_getattr(slot, attno, &isnull);
	if (isnull)
		return false;
	{
		Vector	   *v = DatumGetVector(d);
		Vector	   *copy = (Vector *) palloc(VARSIZE_ANY(v));

		memcpy(copy, v, VARSIZE_ANY(v));
		*out = copy;
	}
	ExecClearTuple(slot);
	return true;
}

static void
recalibrate_image(Relation index, Relation heap, char *blob, int n_new, size_t core_len, TrigpointMetaPageData *meta, int m)
{
	FlatHeader *img = (FlatHeader *) (blob + (size_t) n_new * sizeof(uint64));
	const uint64 *tids = (const uint64 *) blob;
	AttrNumber	attno = index->rd_index->indkey.values[0];
	int			L = meta->L,
				D = meta->D,
				d3 = meta->lv[L - 1];
	void	   *fz;
	int		   *sample;
	float	   *cal[4];
	int			ncal[4] = {0, 0, 0, 0},
				used = 0,
				i,
				l;
	float		ones[4] = {1, 1, 1, 1};
	TupleTableSlot *slot;
	MemoryContext qctx,
				old;
	float	   *head = (float *) palloc(sizeof(float) * (Size) d3);
	float		tail4[4];
	instr_time	t0,
				t1;

	int			probes = (tp_recalibrate_probes > 0 && tp_recalibrate_probes < meta->C) ? tp_recalibrate_probes : meta->C;

	if (m > n_new)
		m = n_new;
	if (m < 1 || n_new <= TP_CAL_K)
		return;
	INSTR_TIME_SET_CURRENT(t0);
	fz = tp_index_open_flat(img, core_len);
	if (fz == NULL)
		ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("trigpoint: cannot open the compacted image for calibration: %s", tp_last_error())));
	sample = (int *) palloc(sizeof(int) * (Size) m);
	m = tp_sample((uint64) meta->generation * 7919 + 2, n_new, m, sample);
	for (l = 0; l < L; l++)
		cal[l] = (float *) palloc(sizeof(float) * (Size) m * TP_CAL_K);
	slot = table_slot_create(heap, NULL);
	qctx = AllocSetContextCreate(CurrentMemoryContext, "trigpoint recalibration query", ALLOCSET_DEFAULT_SIZES);
	PG_TRY();
	{
		for (i = 0; i < m; i++)
		{
			int			t = sample[i];
			Vector	   *q;
			void	   *sc;
			float		tau2 = INFINITY;
			int			top_id[TP_CAL_K + 1];
			float		top_d2[TP_CAL_K + 1];
			int			ntop = 0,
						k;

			MemoryContextReset(qctx);
			old = MemoryContextSwitchTo(qctx);
			if (!fetch_vector(heap, slot, attno, tids[t], &q) || q->dim != D)
			{
				MemoryContextSwitchTo(old);
				continue;
			}
			if (meta->metric == TP_METRIC_COSINE)
				tp_normalize(q->x, D);	/* the slot's detoasted copy in qctx */
			/* the exact top-CAL_K of row t through the new image: certificate scan, exact distances from the heap */
			sc = tp_scan_begin(fz, q->x, 100, probes, 100, ones);
			if (sc == NULL)
				ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("trigpoint: %s", tp_last_error())));
			for (;;)
			{
				int			n = tp_scan_cell(sc, tau2),
							j,
						   *ids;
				float	   *lb2;

				if (n == -1)
					break;
				if (n == -2)
				{
					tp_scan_end(sc);
					ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("trigpoint: %s", tp_last_error())));
				}
				if (n == 0)
					continue;
				ids = (int *) palloc(sizeof(int) * (Size) n);
				lb2 = (float *) palloc(sizeof(float) * (Size) n);
				tp_scan_take(sc, ids, lb2);
				for (j = 0; j < n; j++)
				{
					Vector	   *r;
					float		d2;
					int			pos;

					if (lb2[j] > tau2 * TP_CAL_SLACK)
						break;
					if (ids[j] == t || ids[j] < 0 || ids[j] >= n_new)
						continue;
					if (!fetch_vector(heap, slot, attno, tids[ids[j]], &r) || r->dim != D)
						continue;
					if (meta->metric == TP_METRIC_COSINE)
						tp_normalize(r->x, D);
					d2 = tp_sq_dist(q->x, r->x, D);
					if (ntop == TP_CAL_K && d2 >= top_d2[TP_CAL_K - 1])
						continue;
					pos = (ntop < TP_CAL_K) ? ntop : TP_CAL_K - 1;
					while (pos > 0 && top_d2[pos - 1] > d2)
					{
						top_d2[pos] = top_d2[pos - 1];
						top_id[pos] = top_id[pos - 1];
						pos--;
					}
					top_d2[pos] = d2;
					top_id[pos] = ids[j];
					if (ntop < TP_CAL_K)
						ntop++;
					if (ntop == TP_CAL_K)
						tau2 = top_d2[TP_CAL_K - 1];
				}
				pfree(ids);
				pfree(lb2);
				CHECK_FOR_INTERRUPTS();
			}
			tp_scan_end(sc);
			if (ntop == TP_CAL_K)
			{
				tp_query_sketch(fz, q->x, head, tail4);
				for (k = 0; k < TP_CAL_K; k++)
				{
					float		g[4];

					if (tp_calibrate_pair(fz, head, tail4, top_id[k], tau2, g) < 0)
						continue;
					for (l = 0; l < L; l++)
						cal[l][ncal[l]++] = g[l];
				}
				used++;
			}
			MemoryContextSwitchTo(old);
		}
	}
	PG_CATCH();
	{
		ExecDropSingleTupleTableSlot(slot);
		tp_index_free(fz);
		PG_RE_THROW();
	}
	PG_END_TRY();
	ExecDropSingleTupleTableSlot(slot);
	tp_index_free(fz);
	MemoryContextDelete(qctx);
	if (used >= 20)
	{
		StringInfoData moved;

		initStringInfo(&moved);
		for (l = 0; l < L; l++)
		{
			float	   *dst = FLAT_PTR(img, img->cal[l], float);
			uint64		n = (uint64) ncal[l];
			float		before = img->ch[l];

			if (n > img->cal_cap[l])
				n = img->cal_cap[l];
			memcpy(dst, cal[l], sizeof(float) * (size_t) n);
			if (n > 1)
				qsort(dst, (size_t) n, sizeof(float), tp_cmp_float);
			img->cal_n[l] = n;
			img->ch[l] = n ? tp_ch_from_sorted(dst, (int) n, 0.01) : 1.f;
			meta->ch_default[l] = img->ch[l];
			appendStringInfo(&moved, "%s%.3f->%.3f", l ? "," : "", before, img->ch[l]);
		}
		INSTR_TIME_SET_CURRENT(t1);
		ereport(DEBUG1, (errmsg("trigpoint: re-calibrated \"%s\" over %d of %d sampled live rows in %.0f ms (c_h %s)",
								RelationGetRelationName(index), used, m, INSTR_TIME_GET_MILLISEC(t1) - INSTR_TIME_GET_MILLISEC(t0), moved.data)));
	}
	else
		ereport(DEBUG1, (errmsg("trigpoint: re-calibration of \"%s\" skipped: only %d of %d sampled rows had %d neighbours",
								RelationGetRelationName(index), used, m, TP_CAL_K)));
	for (l = 0; l < L; l++)
		pfree(cal[l]);
	pfree(sample);
	pfree(head);
}

/*
 * amvacuumcleanup: when the log holds more than trigpoint.compact_frac of the image's rows as
 * inserts + tombstones, compact this copy (cells whose overflow + tombstones exceed the fraction
 * are repacked, tombstoned rows disappear), write it as the next generation's image into new
 * pages, and flip the meta page: the log starts empty. Other backends see the new generation on
 * their next sync and reload. The old generation's pages go to the FSM, then E3b moves the image pages
 * above the live page count into them and truncates the file (tp_reclaim_space).
 */
IndexBulkDeleteResult *
tp_vacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats)
{
	Relation	index = info->index;
	TrigpointCached *c;
	TrigpointMetaPageData meta;
	double		churn;
	FlatGraph  *newg = NULL;
	int32	   *newpos = NULL;

	if (stats == NULL)
		stats = (IndexBulkDeleteResult *) palloc0(sizeof(IndexBulkDeleteResult));
	if (info->analyze_only)
	{
		stats->num_pages = RelationGetNumberOfBlocks(index);
		return stats;
	}
	c = tp_cache_get_owned(index);	/* a transient owned copy: the shared image is read-only */
	PG_TRY();
	{
		meta = c->meta;
		/* P5a: the compaction reads every array of the image; a scan guards the values it indexes with, the
		 * compaction runs the deep check once instead (O(n), a small part of the compaction) */
		if (c->priv_image != NULL && meta.core_bytes > 0 &&
			tp_flat_check(c->priv_image + (Size) meta.n * sizeof(uint64), (size_t) meta.core_bytes, 1) < 0)
			ereport(ERROR, (errcode(ERRCODE_INDEX_CORRUPTED),
							errmsg("trigpoint: \"%s\" image: %s", RelationGetRelationName(index), tp_last_error()),
							errhint("Run trigpoint_check(<index>, true); REINDEX rebuilds the index.")));
		/* H13: churn against the live rows (the image holds no tombstones after a renumbering compaction) */
		{
			long long	cnt[3];

			tp_index_counts(c->ix, cnt);	/* total, inserted, deleted */
			churn = (double) (meta.log_inserts + meta.log_deletes) / Max(1.0, (double) (cnt[0] - cnt[2]));
		}
		if (c->n > 0 && (churn > tp_compact_frac || (tp_compact_records > 0 && meta.log_inserts + meta.log_deletes > tp_compact_records)))
		{
			size_t		core_len = 0;
			char	   *blob;
			BlockNumber start;
			Buffer		buf;
			Page		page;
			int			compacted,
						carried = 0,
						freed = 0;
			BlockNumber truncated = 0;
			uint32		moved = 0;
			int			
						total,
						n_new,
						n_old = c->n,
						pos;
			uint64		cal_cap[4];
			TrigpointMetaPageData old;
			long long	counts[3];
			instr_time	t0,
						t1;

			INSTR_TIME_SET_CURRENT(t0);
			/*
			 * The next generation (H13, 2026-10-28): only the live rows, renumbered so that id = position and no
			 * tombstone remains. The router is the image's graph compacted over its live nodes (lists remapped and
			 * repaired, tp_graph_compact), then the log's live rows (this copy's delta graph holds their codes)
			 * inserted at their new positions by the same builder. Before this a compaction kept every position
			 * ever assigned: under churn the image grew without bound and the router's hits were tombstones.
			 */
			tp_index_counts(c->ix, counts);
			total = (int) counts[0];
			newpos = (int32 *) palloc(sizeof(int32) * (Size) Max(total, 1));
			n_new = tp_index_live_map(c->ix, newpos);
			{
				const FlatGraph *base = (const FlatGraph *) tp_index_base_graph(c->ix);
				FlatGraph  *dg = (FlatGraph *) c->dgraph;
				int			k;

				if (base == NULL)
					ereport(ERROR, (errcode(ERRCODE_INDEX_CORRUPTED), errmsg("trigpoint: the compaction copy has no base graph"),
									errhint("Run trigpoint_check() on the index and REINDEX it.")));
				newg = tp_graph_compact(base, newpos, n_new, n_new + 1024);
				if (newg == NULL)
					ereport(ERROR, (errcode(ERRCODE_OUT_OF_MEMORY), errmsg("trigpoint: out of memory for the compacted router graph")));
				compacted = 0;
				if (dg != NULL && dg->n > 0)
				{
					TpGraphCtx	ctx;
					TpGraphScratch *sc = tp_graph_scratch_new(newg);

					ctx.g = newg;
					ctx.sh = NULL;
					for (k = 0; k < dg->n; k++)
					{
						int32		j = (c->n_base + k < total) ? newpos[c->n_base + k] : -1;

						if (j < 0 || fg_level_c(dg)[k] < 0)
							continue;	/* tombstoned, or a cancelled pair the replay never graphed */
						tp_graph_set_code(newg, j, fg_code(dg, k));
						tp_graph_insert(&ctx, j, sc);
						compacted++;
						if ((k & 1023) == 0)
							CHECK_FOR_INTERRUPTS();
					}
					tp_graph_scratch_free(sc);
				}
				newg->n = n_new;
			}
			/* room for the re-calibration's tables (H16): m queries x CAL_K pairs per level */
			for (pos = 0; pos < 4; pos++)
				cal_cap[pos] = (uint64) Min(tp_recalibrate_queries, n_new) * TP_CAL_K;
			{
				long long	flat = tp_index_reflatten_live(c->ix, newg, newpos, n_new, cal_cap, NULL, 0);

				if (flat < 0)
					ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("trigpoint: %s", tp_last_error())));
				core_len = (size_t) flat;
			}
			meta.generation++;
			meta.n = n_new;			/* ids 0..n_new-1 = the live rows in their old position order */
			meta.n_image = n_new;
			meta.core_bytes = core_len;
			meta.blob_bytes = (uint64) n_new * sizeof(uint64) + core_len;
			meta.log_first = meta.log_last = InvalidBlockNumber;
			meta.log_inserts = meta.log_deletes = 0;
			meta.compactions++;
			blob = (char *) malloc(meta.blob_bytes);
			if (blob == NULL)
				ereport(ERROR, (errcode(ERRCODE_OUT_OF_MEMORY), errmsg("trigpoint: out of memory for the index image")));
			for (pos = 0; pos < total; pos++)
			{
				ItemPointerData tid;
				uint64		t;
				int32		j = newpos[pos];
				int			id;

				if (j < 0)
					continue;
				id = tp_index_pos_id(c->ix, pos);
				if (id < 0 || !tp_tid_of(c, id, &tid) || tp_index_is_deleted(c->ix, id))
					ereport(ERROR, (errcode(ERRCODE_INDEX_CORRUPTED),
									errmsg("trigpoint: live position %d (id %d) of \"%s\" has no heap TID", pos, id, RelationGetRelationName(index)),
									errhint("Run trigpoint_check() on the index and REINDEX it.")));
				t = ((uint64) ItemPointerGetBlockNumber(&tid) << 16) | (uint64) ItemPointerGetOffsetNumber(&tid);
				memcpy(blob + (size_t) j * sizeof(uint64), &t, sizeof(uint64));
			}
			if (tp_index_reflatten_live(c->ix, newg, newpos, n_new, cal_cap, blob + (size_t) n_new * sizeof(uint64), core_len) != (long long) core_len)
				ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("trigpoint: %s", tp_last_error())));
			free(newg);
			newg = NULL;
			if (tp_recalibrate_queries > 0 && info->heaprel != NULL)
				recalibrate_image(index, info->heaprel, blob, n_new, core_len, &meta, tp_recalibrate_queries);
			start = InvalidBlockNumber;
			tp_write_image_pages(index, MAIN_FORKNUM, &meta, blob, &start);	/* pages from the FSM or the end, logged one by one */
			free(blob);
			meta.image_start = start;
			/* flip: readers hold the meta page's share lock while they load and appenders its exclusive
			 * lock, so this waits for them; records appended since our sync are carried into the new
			 * generation's log first (tp_carry_log), so no committed insert or tombstone is lost */
			buf = ReadBuffer(index, TP_METAPAGE_BLKNO);
			LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
			{
				TrigpointMetaPageData cur;
				GenericXLogState *state;

				memcpy(&cur, TrigpointPageGetMeta(BufferGetPage(buf)), sizeof(cur));
				if (cur.generation != meta.generation - 1)
					elog(ERROR, "trigpoint: generation changed under the compaction of \"%s\"", RelationGetRelationName(index));
				old = cur;
				carried = tp_carry_log(index, c, &meta, &cur, newpos, n_old);
				state = GenericXLogStart(index);
				page = GenericXLogRegisterBuffer(state, buf, 0);
				memcpy(TrigpointPageGetMeta(page), &meta, sizeof(meta));
				GenericXLogFinish(state);
			}
			UnlockReleaseBuffer(buf);
			tp_shared_retire(index, meta.generation - 1);	/* the old generation's segment goes when its last reader detaches */
			freed = tp_free_generation_pages(index, &old);	/* E3: its image and log pages return to the FSM */
			if (tp_compact_reclaim)
				truncated = tp_reclaim_space(index, &meta, &moved);	/* E3b: the image into them, then truncate */
			c->meta = meta;
			c->generation = meta.generation;
			c->log_blk = InvalidBlockNumber;
			c->log_off = InvalidOffsetNumber;
			INSTR_TIME_SET_CURRENT(t1);
			ereport(DEBUG1, (errmsg("trigpoint: compacted \"%s\": %d live rows of %d positions, %d log rows inserted into the router, generation %u, %u image pages, %d log records carried, %d pages freed, %u image pages moved down, %u pages truncated, %.0f ms",
									RelationGetRelationName(index), n_new, total, compacted, meta.generation, meta.nchunks, carried, freed, moved, truncated,
									INSTR_TIME_GET_MILLISEC(t1) - INSTR_TIME_GET_MILLISEC(t0))));
			pfree(newpos);
			newpos = NULL;
		}
	}
	PG_CATCH();
	{
		if (newg)
			free(newg);
		tp_cache_unpin(c);
		PG_RE_THROW();
	}
	PG_END_TRY();
	stats->num_index_tuples = c->n;
	tp_cache_unpin(c);			/* frees the transient copy */
	stats->num_pages = RelationGetNumberOfBlocks(index);
	return stats;
}
