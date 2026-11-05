/* Trigpoint access method: the read path.
 *
 * Per-backend cache (trigpoint.h TrigpointCached): the first scan of an index in a backend opens it —
 * shared (E1, default): maps the generation's flat image from its DSM segment (tp_shared.c; the first
 * backend in the cluster to need a generation loads it from the pages) and the core reads it in
 * place, keeping only the log since the image private; owned (shared_image = off, and VACUUM's
 * compaction): reads blocks image_start.. into a malloc'd buffer and deserialises its own core index.
 * The entry lives until the relcache entry of the index is invalidated (REINDEX, DROP, ANALYZE ...)
 * or the backend exits.
 *
 * amgettuple: on the first call a top-k batch is computed — route (project, beam, rank cells),
 * then for each of the P cells the core returns the level-3 survivors with their lower bounds and
 * this file fetches their heap rows (table_index_fetch_tuple with the scan's snapshot: dead and
 * invisible rows are skipped), computes the exact distance and tightens tau — then the TIDs are
 * handed out in distance order. When the executor asks for more (a WHERE above the index, a LIMIT
 * above trigpoint.k) the scan goes on in batches (trigpoint.iterative_scan): k and the cells opened
 * double, the core keeps the routed order and re-cascades the opened cells under the larger tau,
 * and no heap row is fetched twice. No PostgreSQL call runs while a core frame is on the stack.
 */
#include "trigpoint.h"

#include <float.h>
#include <math.h>

#include "access/generic_xlog.h"
#include "access/tableam.h"
#include "catalog/pg_type.h"
#include "executor/tuptable.h"
#include "miscadmin.h"
#include "optimizer/cost.h"
#include "optimizer/optimizer.h"
#include "storage/bufmgr.h"
#include "utils/inval.h"
#include "utils/hsearch.h"
#include "utils/memutils.h"
#include "storage/lmgr.h"
#include "utils/selfuncs.h"
#include "portability/instr_time.h"

#include "extension/vector/vector.h"

#include "tp_graph.h"

/* ------------------------------------------------------------------------------------------------ */
/* per-backend cache */

#define TP_CACHE_MAX 16
static TrigpointCached *tp_cache[TP_CACHE_MAX];
static int	tp_cache_n = 0;
static bool tp_cache_callback_registered = false;

/* release the core index, the segment mapping and the TID table; the entry itself stays */
static void
tp_cache_reset(TrigpointCached *c)
{
	if (c->ix)
		tp_index_free(c->ix);
	c->ix = NULL;
	tp_delta_graph_free(c);
	tp_shared_release(c);
	if (c->priv_image)
		free(c->priv_image);
	c->priv_image = NULL;
	free(c->tids);
	c->tids = NULL;
	c->n_cap = 0;
	c->n = 0;
}

static void
tp_cache_free(TrigpointCached *c)
{
	tp_cache_reset(c);
	pfree(c);
}

static void
tp_cache_drop(int i)
{
	TrigpointCached *c = tp_cache[i];

	tp_cache[i] = tp_cache[--tp_cache_n];
	if (c->pins > 0)
		c->stale = true;		/* a running scan still uses it: freed at its unpin */
	else
		tp_cache_free(c);
}

static void
tp_relcache_callback(Datum arg, Oid relid)
{
	int			i;

	for (i = tp_cache_n - 1; i >= 0; i--)
		if (relid == InvalidOid || tp_cache[i]->relid == relid)
			tp_cache_drop(i);
}

/*
 * The structural checks every reader of a meta page runs before trusting a field (H3): magic, format, the
 * counts, the level widths, and the image sizes the loaders copy by. Returns NULL when the page passes, else
 * a palloc'd one-line reason. trigpoint_check reports the same reason as a finding.
 */
char *
tp_meta_problem(const TrigpointMetaPageData *meta, BlockNumber nblocks)
{
	int			l;

	if (meta->magic != TP_MAGIC)
		return psprintf("not a trigpoint meta page (magic 0x%08x)", meta->magic);
	if (meta->version != TP_META_VERSION)
		return psprintf("page format %u, this version reads %d (REINDEX)", meta->version, TP_META_VERSION);
	if (meta->n == 0 && meta->n_image == 0 && meta->core_bytes == 0 && meta->D >= 1)
		return NULL;			/* a model-less page written before H14 (0.2's ambuildempty): nothing else to check */
	if (meta->n < 0 || meta->D < 1 || meta->C < 1 || meta->L < 1 || meta->L > 4 || meta->n_image != meta->n)
		return psprintf("counts out of range (n=%d n_image=%d D=%d cells=%d levels=%d)", meta->n, meta->n_image, meta->D, meta->C, meta->L);
	for (l = 0; l < meta->L; l++)
		if (meta->lv[l] < 1 || meta->lv[l] > meta->D || (l > 0 && meta->lv[l] <= meta->lv[l - 1]))
			return psprintf("level widths not increasing within [1, D=%d]", meta->D);
	if (meta->core_bytes > 0 && (meta->blob_bytes != (uint64) meta->n * sizeof(uint64) + meta->core_bytes || meta->core_bytes < 64 ||
						meta->nchunks == 0 || (uint64) meta->nchunks * TP_CHUNK_BYTES < meta->blob_bytes ||
						!BlockNumberIsValid(meta->image_start) || meta->image_start == TP_METAPAGE_BLKNO || meta->image_start >= nblocks))
		return psprintf("image sizes inconsistent (n=%d image=" UINT64_FORMAT " core=" UINT64_FORMAT " pages=%u start=%u of %u)",
						meta->n, meta->blob_bytes, meta->core_bytes, meta->nchunks, meta->image_start, nblocks);
	if ((BlockNumberIsValid(meta->log_first) != BlockNumberIsValid(meta->log_last)) ||
		(BlockNumberIsValid(meta->log_first) && (meta->log_first >= nblocks || meta->log_last >= nblocks ||
												  meta->log_first == TP_METAPAGE_BLKNO || meta->log_last == TP_METAPAGE_BLKNO)))
		return psprintf("log chain pointers out of range (first=%u last=%u of %u)", meta->log_first, meta->log_last, nblocks);
	if (meta->log_inserts < 0 || meta->log_deletes < 0)
		return psprintf("negative log counters");
	return NULL;
}

void
tp_read_meta(Relation index, TrigpointMetaPageData *meta)
{
	Buffer		buf = ReadBuffer(index, TP_METAPAGE_BLKNO);
	char	   *why;

	LockBuffer(buf, BUFFER_LOCK_SHARE);
	memcpy(meta, TrigpointPageGetMeta(BufferGetPage(buf)), sizeof(TrigpointMetaPageData));
	UnlockReleaseBuffer(buf);
	why = tp_meta_problem(meta, RelationGetNumberOfBlocks(index));
	if (why != NULL)
		ereport(ERROR, (errcode(ERRCODE_INDEX_CORRUPTED),
						errmsg("trigpoint: \"%s\": %s", RelationGetRelationName(index), why),
						errhint("Run trigpoint_check('%s'); REINDEX rebuilds the index.", RelationGetRelationName(index))));
}

/* ------------------------------------------------------------------------------------------------ */
/* the delta graph: this backend's router over the positions appended since the image (tp_graph.c) */

void
tp_delta_graph_free(TrigpointCached *c)
{
	if (c->dgraph)
		free(c->dgraph);
	c->dgraph = NULL;
	if (c->dscratch)
		tp_graph_scratch_free((TpGraphScratch *) c->dscratch);
	c->dscratch = NULL;
	if (c->ix)
		tp_index_set_delta_graph(c->ix, NULL);
}

/* insert delta node k (= position n_base + k) with its codes; grows the graph as needed */
void
tp_delta_graph_add(TrigpointCached *c, int k, const int8 *codes)
{
	FlatGraph  *g = (FlatGraph *) c->dgraph;
	TpGraphCtx	ctx;

	if (g == NULL)
	{
		int			prm[3];
		const float *step;
		MemoryContext old;

		tp_index_router_params(c->ix, prm, &step);
		g = tp_graph_private_new(Max(1024, 2 * (k + 1)), prm[0], prm[1], prm[2], step, 0x7472696770ULL /* "trigp" */);
		if (g == NULL)
			ereport(ERROR, (errcode(ERRCODE_OUT_OF_MEMORY), errmsg("trigpoint: out of memory for the delta graph")));
		c->dgraph = g;
		old = MemoryContextSwitchTo(TopMemoryContext);
		c->dscratch = tp_graph_scratch_new(g);
		MemoryContextSwitchTo(old);
	}
	else if (k >= g->n_cap)
	{
		FlatGraph  *ng = tp_graph_private_grow(g, Max(2 * g->n_cap, k + 1));

		if (ng == NULL)
			ereport(ERROR, (errcode(ERRCODE_OUT_OF_MEMORY), errmsg("trigpoint: out of memory growing the delta graph")));
		c->dgraph = g = ng;
	}
	tp_graph_set_code(g, k, codes);
	fg_level(g)[k] = 0;			/* present; tp_graph_insert draws its level when it links it */
	if (k + 1 > g->n)
		g->n = k + 1;
	ctx.g = g;
	ctx.sh = NULL;
	/*
	 * H17 (2026-10-29): linking costs ~0.45 ms a node at 1M x 1536 (a fresh backend replaying 20,000 records
	 * waited 9 s for its first query), while the router scores an unlinked delta flat in well under a
	 * millisecond. So the nodes stay unlinked (enterpoint < 0) until the delta outgrows
	 * trigpoint.delta_flat_max; then every present node is linked once and later ones as they arrive.
	 */
	if (g->enterpoint >= 0)
		tp_graph_insert(&ctx, k, (TpGraphScratch *) c->dscratch);
	else if (g->n > tp_delta_flat_max)
	{
		int			i;

		for (i = 0; i < g->n; i++)
			if (fg_level(g)[i] >= 0)
				tp_graph_insert(&ctx, i, (TpGraphScratch *) c->dscratch);
	}
	tp_index_set_delta_graph(c->ix, g);	/* the pointer may have moved */
}

/* the heap TID of an id: the segment's table for ids below n_base, this backend's table above */
bool
tp_tid_of(TrigpointCached *c, int id, ItemPointer out)
{
	if (id < 0 || id >= c->n)
		return false;
	if (id < c->n_base)
	{
		uint64		t = c->tids_base[id];

		if (t == 0)
			return false;		/* tombstoned before the image was written */
		ItemPointerSet(out, (BlockNumber) (t >> 16), (OffsetNumber) (t & 0xFFFF));
		return true;
	}
	*out = c->tids[id - c->n_base];
	return ItemPointerIsValid(out);
}

/* grow this backend's TID table to hold n ids (beyond n_base) */
static void
tids_reserve(TrigpointCached *c, int n)
{
	if (n <= c->n_cap)
		return;
	while (c->n_cap < n)
		c->n_cap = c->n_cap ? c->n_cap * 2 : 1024;
	c->tids = (ItemPointerData *) realloc(c->tids, sizeof(ItemPointerData) * (size_t) c->n_cap);
	if (c->tids == NULL)
		ereport(ERROR, (errcode(ERRCODE_OUT_OF_MEMORY), errmsg("trigpoint: out of memory for the TID table")));
}

/*
 * The record at (page, off), checked before anything reads through it (H3): the item is at least a record
 * header, its length is the length its kind implies for this index, the cell is in range and the norms are
 * finite. tp_log_record_problem never throws (trigpoint_check calls it under a buffer lock and reports the
 * reason); tp_log_record is the replay's form and raises ERROR (index corrupted) with the block number.
 */
TrigpointLogRecord *
tp_log_record_problem(Page page, OffsetNumber off, BlockNumber blk, int L, const int32 *lv, int C, char **why)
{
	ItemId		iid = PageGetItemId(page, off);
	TrigpointLogRecord *rec;
	Size		len = ItemIdGetLength(iid),
				want;
	int			i;

	*why = NULL;
	if (!tp_item_fits(page, iid) || len < offsetof(TrigpointLogRecord, codes))
	{
		*why = psprintf("log record at block %u offset %u is not a whole item of its page or too short (%zu bytes)", blk, off, len);
		return NULL;
	}
	rec = (TrigpointLogRecord *) PageGetItem(page, iid);
	if (rec->kind == TP_REC_INSERT)
		want = TP_INSERT_RECORD_SIZE(lv[L - 1]);
	else if (rec->kind == TP_REC_DELETE)
		want = TP_DELETE_RECORD_SIZE;
	else
	{
		*why = psprintf("log record at block %u offset %u has kind %u", blk, off, rec->kind);
		return NULL;
	}
	if (len != want)
	{
		*why = psprintf("log record at block %u offset %u is %zu bytes, kind %u needs %zu", blk, off, len, rec->kind, want);
		return NULL;
	}
	if (rec->kind == TP_REC_INSERT)
	{
		if (rec->cell >= C)
			*why = psprintf("log record at block %u offset %u names cell %u of %d", blk, off, rec->cell, C);
		else if (rec->id < 0)
			*why = psprintf("log record at block %u offset %u has id %d", blk, off, rec->id);
		else if (rec->tid == 0 || (rec->tid & 0xFFFF) == 0)
			*why = psprintf("log record at block %u offset %u has an invalid heap TID", blk, off);
		else
			for (i = 0; i < 2 * L; i++)
				if (!isfinite(rec->norms[i]) || rec->norms[i] < 0)
				{
					*why = psprintf("log record at block %u offset %u has a norm that is negative or not finite", blk, off);
					break;
				}
	}
	else if (rec->id < 0)
		*why = psprintf("tombstone at block %u offset %u has id %d", blk, off, rec->id);
	return *why ? NULL : rec;
}

TrigpointLogRecord *
tp_log_record(Page page, OffsetNumber off, BlockNumber blk, int D, int L, const int32 *lv, int C)
{
	char	   *why;
	TrigpointLogRecord *rec = tp_log_record_problem(page, off, blk, L, lv, C, &why);

	(void) D;
	if (rec == NULL)
		ereport(ERROR, (errcode(ERRCODE_INDEX_CORRUPTED), errmsg("trigpoint: %s", why)));
	return rec;
}

/* apply one log record to this copy (inserts get id = next id, which must equal rec->id) */
void
tp_cache_apply_record(TrigpointCached *c, TrigpointLogRecord *rec)
{
	if (rec->kind == TP_REC_INSERT)
	{
		int			id = c->n;
		int			pos;

		if (rec->id != id)
			ereport(ERROR, (errcode(ERRCODE_INDEX_CORRUPTED), errmsg("trigpoint: log record id %d does not match the next id %d (log out of order?)", rec->id, id), errhint("Run trigpoint_check(<index>, true); REINDEX rebuilds the index.")));
		pos = tp_index_insert(c->ix, id, rec->cell, rec->codes, rec->norms, rec->norms + 4);
		if (pos < 0)
			ereport(ERROR, (errcode(ERRCODE_INDEX_CORRUPTED), errmsg("trigpoint: log record %d cannot be applied: %s", rec->id, tp_last_error()), errhint("Run trigpoint_check(<index>, true); REINDEX rebuilds the index.")));
		/* A record whose DELETE is ahead in the same replay gets its position and its tombstone but no graph
		 * node (≈ 1 ms each: 43,238 records took 10–22 s at connect on the 1M table). The core's live count
		 * does not see it either, so the delta graph is searched only for inserts that are still live. */
		if (c->skip_dead != NULL && id >= c->skip_base && id - c->skip_base < c->skip_n && c->skip_dead[id - c->skip_base])
			c->skipped++;
		else
			tp_delta_graph_add(c, pos - c->n_base, rec->codes);
		tids_reserve(c, id + 1 - c->n_base);
		ItemPointerSet(&c->tids[id - c->n_base], (BlockNumber) (rec->tid >> 16), (OffsetNumber) (rec->tid & 0xFFFF));
		c->n = id + 1;
	}
	else if (rec->kind == TP_REC_DELETE)
	{
		/* a tombstone names an id already assigned (its INSERT precedes it in the log, or it is in the image) */
		if (rec->id < 0 || rec->id >= c->n)
			ereport(ERROR, (errcode(ERRCODE_INDEX_CORRUPTED), errmsg("trigpoint: tombstone for id %d but only %d ids exist (log out of order?)", rec->id, c->n)));
		tp_index_delete(c->ix, &rec->id, 1);	/* the core's tombstone keeps the id out of every scan */
		if (rec->id >= c->n_base)
			ItemPointerSetInvalid(&c->tids[rec->id - c->n_base]);
	}
	else
		ereport(ERROR, (errcode(ERRCODE_INDEX_CORRUPTED), errmsg("trigpoint: unknown log record kind %u", rec->kind), errhint("Run trigpoint_check(<index>, true); REINDEX rebuilds the index.")));
	c->replayed++;
}

/* apply the records of one log page beyond offset `from` (exclusive); returns the last offset applied */
static OffsetNumber
apply_log_page(TrigpointCached *c, Page page, BlockNumber blk, OffsetNumber from)
{
	OffsetNumber max = PageGetMaxOffsetNumber(page), off;

	for (off = (from == InvalidOffsetNumber) ? FirstOffsetNumber : from + 1; off <= max; off++)
	{
		ItemId		iid = PageGetItemId(page, off);

		if (!ItemIdIsUsed(iid))
			continue;
		tp_cache_apply_record(c, tp_log_record(page, off, blk, c->D, c->L, c->meta.lv, c->C));
	}
	return max;
}

/* read and check one log page of this generation; returns the pinned, share-locked buffer */
static Buffer
read_log_page(Relation index, BlockNumber blk, uint32 generation, BlockNumber *nblocks)
{
	Buffer		buf = tp_read_linked(index, blk, nblocks);
	TrigpointPageOpaque op;

	LockBuffer(buf, BUFFER_LOCK_SHARE);
	op = tp_page_header_ok(BufferGetPage(buf)) ? TrigpointPageGetOpaque(BufferGetPage(buf)) : NULL;
	if (op == NULL || op->kind != TP_PAGE_LOG || op->generation != generation)
	{
		UnlockReleaseBuffer(buf);
		ereport(ERROR, (errcode(ERRCODE_INDEX_CORRUPTED), errmsg("trigpoint: block %u is not a log page of generation %u", blk, generation), errhint("Run trigpoint_check(<index>, true); REINDEX rebuilds the index.")));
	}
	return buf;
}

/*
 * Pass over the records ahead of (blk, off) without applying them: count the INSERT records and collect
 * the ids >= n0 their DELETE records cancel. Returns a palloc0'd byte per INSERT ahead (1 = cancelled) in
 * *dead, or NULL when nothing is cancelled.
 */
static void
prescan_log(Relation index, TrigpointCached *c, const TrigpointMetaPageData *meta, BlockNumber blk, OffsetNumber from,
			int *ninserts, uint8 **dead)
{
	int			n0 = c->n,
				ndead = 0,
				cap = 0,
				i;
	int32	   *dead_ids = NULL;
	BlockNumber nblocks = 0;

	*ninserts = 0;
	*dead = NULL;
	for (;;)
	{
		Buffer		buf = read_log_page(index, blk, meta->generation, &nblocks);
		Page		page = BufferGetPage(buf);
		OffsetNumber max = PageGetMaxOffsetNumber(page),
					off;
		BlockNumber next = TrigpointPageGetOpaque(page)->next;

		for (off = (from == InvalidOffsetNumber) ? FirstOffsetNumber : from + 1; off <= max; off++)
		{
			ItemId		iid = PageGetItemId(page, off);
			TrigpointLogRecord *rec;

			if (!ItemIdIsUsed(iid))
				continue;
			rec = tp_log_record(page, off, blk, c->D, c->L, meta->lv, c->C);
			if (rec->kind == TP_REC_INSERT)
				(*ninserts)++;
			else if (rec->kind == TP_REC_DELETE && rec->id >= n0)
			{
				if (ndead == cap)
				{
					cap = cap ? cap * 2 : 1024;
					dead_ids = dead_ids ? (int32 *) repalloc(dead_ids, sizeof(int32) * (Size) cap)
						: (int32 *) palloc(sizeof(int32) * (Size) cap);
				}
				dead_ids[ndead++] = rec->id;
			}
		}
		UnlockReleaseBuffer(buf);
		if (!BlockNumberIsValid(next))
			break;
		blk = next;
		from = InvalidOffsetNumber;
		CHECK_FOR_INTERRUPTS();
	}
	if (ndead > 0 && *ninserts > 0)
	{
		*dead = (uint8 *) palloc0((Size) *ninserts);
		for (i = 0; i < ndead; i++)
			if (dead_ids[i] - n0 < *ninserts)
				(*dead)[dead_ids[i] - n0] = 1;
	}
	if (dead_ids)
		pfree(dead_ids);
}

/*
 * Bring the copy up to date: a new generation means reload (the meta page's share lock is held
 * by the caller for the whole load); otherwise apply the log records appended since the last sync,
 * following the page chain from where we stopped. With skip_cancelled (the load's replay of a whole
 * log) INSERT records whose DELETE follows in the same pass are not inserted into the delta graph.
 */
static void
apply_log_from(Relation index, TrigpointCached *c, const TrigpointMetaPageData *meta, bool skip_cancelled)
{
	BlockNumber blk;
	uint8	   *dead = NULL;
	int			ninserts = 0;
	BlockNumber nblocks = 0;

	if (!BlockNumberIsValid(meta->log_first))
		return;
	blk = BlockNumberIsValid(c->log_blk) ? c->log_blk : meta->log_first;
	if (skip_cancelled)
		prescan_log(index, c, meta, blk, blk == c->log_blk ? c->log_off : InvalidOffsetNumber, &ninserts, &dead);
	c->skip_dead = dead;
	c->skip_base = c->n;
	c->skip_n = dead ? ninserts : 0;
	PG_TRY();
	{
		for (;;)
		{
			Buffer		buf = read_log_page(index, blk, meta->generation, &nblocks);
			BlockNumber next = TrigpointPageGetOpaque(BufferGetPage(buf))->next;

			c->log_off = apply_log_page(c, BufferGetPage(buf), blk, blk == c->log_blk ? c->log_off : InvalidOffsetNumber);
			c->log_blk = blk;
			UnlockReleaseBuffer(buf);
			if (!BlockNumberIsValid(next))
				break;
			blk = next;
			CHECK_FOR_INTERRUPTS();
		}
	}
	PG_FINALLY();
	{
		c->skip_dead = NULL;
		c->skip_n = 0;
	}
	PG_END_TRY();
	if (dead)
		pfree(dead);
}

/*
 * Load the generation `meta` describes into an empty entry: shared (map the segment, open the flat
 * image in place) or owned (read the pages into this backend's core index), then replay the log.
 */
static void
load_image(Relation index, TrigpointCached *c, const TrigpointMetaPageData *meta)
{
	instr_time	t0,
				t1;
	TrigpointSegHeader *h;

	c->meta = *meta;
	c->n = meta->n;
	c->D = meta->D;
	c->C = meta->C;
	c->L = meta->L;
	c->generation = meta->generation;
	c->log_blk = InvalidBlockNumber;
	c->log_off = InvalidOffsetNumber;
	c->t_read = c->t_deser = c->t_flatten = c->t_attach = c->t_replay = 0;
	c->replayed = 0;
	c->skipped = 0;
	c->skip_dead = NULL;
	c->skip_n = 0;
	c->loaded_here = false;
	if (meta->core_bytes == 0)
		return;					/* a model-less index (0.2's ambuildempty): nothing to load, inserts are refused */
	/* Both forms open the flat image in place (frozen core index); they differ in whose memory holds it. */
	if (!c->owned && tp_shared_effective() && (h = tp_shared_acquire(index, meta, c)) != NULL)
	{
		INSTR_TIME_SET_CURRENT(t0);
		c->tids_base = (const uint64 *) ((char *) h + h->tids_off);
		c->n_base = h->n;
		c->ix = tp_index_open_flat((char *) h + h->flat_off, (size_t) h->flat_bytes);
		if (c->ix == NULL)
		{
			tp_shared_release(c);
			ereport(ERROR, (errcode(ERRCODE_INDEX_CORRUPTED), errmsg("trigpoint: cannot open the shared image of \"%s\": %s",
																	  RelationGetRelationName(index), tp_last_error())));
		}
		INSTR_TIME_SET_CURRENT(t1);
		c->t_attach += INSTR_TIME_GET_MILLISEC(t1) - INSTR_TIME_GET_MILLISEC(t0);
	}
	else
	{
		/* owned, no registry (16 without the preload), or the cap under on_shared_cap = private: this backend's copy */
		INSTR_TIME_SET_CURRENT(t0);
		c->priv_image = tp_read_blob(index, meta);	/* TIDs then the flat image, this backend's copy */
		INSTR_TIME_SET_CURRENT(t1);
		c->t_read = INSTR_TIME_GET_MILLISEC(t1) - INSTR_TIME_GET_MILLISEC(t0);
		c->tids_base = (const uint64 *) c->priv_image;
		c->n_base = meta->n;
		c->ix = tp_index_open_flat(c->priv_image + (size_t) meta->n * sizeof(uint64), (size_t) meta->core_bytes);
		if (c->ix == NULL)
			ereport(ERROR, (errcode(ERRCODE_INDEX_CORRUPTED), errmsg("trigpoint: cannot open the image of \"%s\": %s",
																	  RelationGetRelationName(index), tp_last_error())));
		INSTR_TIME_SET_CURRENT(t0);
		c->t_deser = INSTR_TIME_GET_MILLISEC(t0) - INSTR_TIME_GET_MILLISEC(t1);	/* open: no deserialisation any more */
	}
	INSTR_TIME_SET_CURRENT(t0);
	apply_log_from(index, c, meta, true);
	INSTR_TIME_SET_CURRENT(t1);
	c->t_replay = INSTR_TIME_GET_MILLISEC(t1) - INSTR_TIME_GET_MILLISEC(t0);
	if (tp_debug)
		ereport(NOTICE, (errmsg("trigpoint: %s \"%s\" generation %u%s: read %.1f ms, deserialise %.1f ms, flatten %.1f ms, attach %.2f ms, "
								"replay %d records %.2f ms (%d cancelled, not graphed), private %lld bytes",
								c->seg ? "shared" : "owned", RelationGetRelationName(index), meta->generation,
								c->loaded_here ? " (loaded the segment)" : "", c->t_read, c->t_deser, c->t_flatten, c->t_attach,
								c->replayed, c->t_replay, c->skipped, (long long) tp_index_private_bytes(c->ix))));
}

/* Sync an existing copy with the index: reload on a new generation, else apply new log records. */
void
tp_cache_sync(Relation index, TrigpointCached *c)
{
	Buffer		mbuf = ReadBuffer(index, TP_METAPAGE_BLKNO);
	TrigpointMetaPageData meta;

	LockBuffer(mbuf, BUFFER_LOCK_SHARE);
	memcpy(&meta, TrigpointPageGetMeta(BufferGetPage(mbuf)), sizeof(meta));
	{
		char	   *why = tp_meta_problem(&meta, RelationGetNumberOfBlocks(index));

		if (why != NULL)
			ereport(ERROR, (errcode(ERRCODE_INDEX_CORRUPTED), errmsg("trigpoint: \"%s\": %s", RelationGetRelationName(index), why),
							errhint("Run trigpoint_check('%s'); REINDEX rebuilds the index.", RelationGetRelationName(index))));
	}
	/*
	 * No PG_TRY here: an error below (a corrupt page, out of memory) is cleaned up by the transaction abort, which
	 * releases the buffer's lock and pin itself. Releasing them from a PG_CATCH would run LWLockRelease after
	 * errfinish has zeroed InterruptHoldoffCount, underflowing it (the backend then ignores every interrupt; the
	 * cassert build asserts in lwlock.c). The cache entry is left for tp_cache_new's / the caller's cleanup.
	 */
	if (meta.generation != c->generation)
	{
		tp_cache_reset(c);
		load_image(index, c, &meta);
	}
	else if (BlockNumberIsValid(meta.log_last) &&
			 (meta.log_last != c->log_blk || meta.log_inserts + meta.log_deletes > 0))
	{
		c->meta = meta;
		apply_log_from(index, c, &meta, false);
	}
	c->meta = meta;
	UnlockReleaseBuffer(mbuf);
}

static TrigpointCached *
tp_cache_new(Relation index, bool owned)
{
	TrigpointCached *c = (TrigpointCached *) MemoryContextAllocZero(TopMemoryContext, sizeof(TrigpointCached));

	c->relid = RelationGetRelid(index);
	c->owned = owned;
	c->generation = (uint32) -1;
	c->log_blk = InvalidBlockNumber;
	c->log_off = InvalidOffsetNumber;
	PG_TRY();
	{
		tp_cache_sync(index, c);	/* generation differs -> load */
	}
	PG_CATCH();
	{
		tp_cache_free(c);
		PG_RE_THROW();
	}
	PG_END_TRY();
	c->pins = 1;
	return c;
}

TrigpointCached *
tp_cache_get(Relation index)
{
	Oid			relid = RelationGetRelid(index);
	TrigpointCached *c;
	int			i;

	for (i = 0; i < tp_cache_n; i++)
		if (tp_cache[i]->relid == relid)
		{
			if (tp_cache[i]->owned == tp_shared_effective())
			{
				tp_cache_drop(i);	/* trigpoint.shared_image changed since this entry was opened */
				break;
			}
			tp_cache[i]->pins++;
			PG_TRY();
			{
				tp_cache_sync(index, tp_cache[i]);
			}
			PG_CATCH();
			{
				tp_cache[i]->pins--;
				PG_RE_THROW();
			}
			PG_END_TRY();
			return tp_cache[i];
		}

	if (!tp_cache_callback_registered)
	{
		CacheRegisterRelcacheCallback(tp_relcache_callback, (Datum) 0);
		tp_cache_callback_registered = true;
	}
	c = tp_cache_new(index, !tp_shared);
	if (tp_cache_n == TP_CACHE_MAX)
		tp_cache_drop(0);
	tp_cache[tp_cache_n++] = c;
	return c;
}

/* A transient owned copy for the compaction (never shared, not cached: freed at its unpin). */
TrigpointCached *
tp_cache_get_owned(Relation index)
{
	TrigpointCached *c = tp_cache_new(index, true);

	c->stale = true;
	return c;
}

/*
 * Append one record to the log (inserts and tombstones). Under the exclusive lock of the tail
 * page: first apply any records other backends appended since our last sync (so our id for an
 * insert is the next one), then add ours, extending the chain with a new page when it is full
 * (meta and the old tail are updated in the same generic-xlog record), then apply it locally.
 */
void
tp_log_append(Relation index, TrigpointCached *c, TrigpointLogRecord *rec, Size size)
{
	Buffer		mbuf,
				tbuf = InvalidBuffer,
				nbuf = InvalidBuffer;
	Page		mpage,
				tpage;
	TrigpointMetaPage meta;
	GenericXLogState *state;
	OffsetNumber off;

	/* the meta page's exclusive lock serialises appenders (and excludes a compaction flip) */
	mbuf = ReadBuffer(index, TP_METAPAGE_BLKNO);
	LockBuffer(mbuf, BUFFER_LOCK_EXCLUSIVE);
	mpage = BufferGetPage(mbuf);
	meta = TrigpointPageGetMeta(mpage);
	if (meta->generation != c->generation)
	{
		/* a compaction happened since we looked: reload under this lock (nobody can flip now) */
		TrigpointMetaPageData m = *meta;

		tp_cache_reset(c);
		load_image(index, c, &m);
	}
	else if (BlockNumberIsValid(meta->log_last))
	{
		TrigpointMetaPageData m = *meta;

		apply_log_from(index, c, &m, false);	/* records appended by others since our sync */
	}
	c->meta = *meta;
	if (rec->kind == TP_REC_INSERT)
		rec->id = c->n;

	if (BlockNumberIsValid(meta->log_last))
	{
		tbuf = ReadBuffer(index, meta->log_last);
		LockBuffer(tbuf, BUFFER_LOCK_EXCLUSIVE);
		tpage = BufferGetPage(tbuf);
		if (PageGetFreeSpace(tpage) < size + sizeof(ItemIdData))
		{
			UnlockReleaseBuffer(tbuf);
			tbuf = InvalidBuffer;
		}
	}
	state = GenericXLogStart(index);
	mpage = GenericXLogRegisterBuffer(state, mbuf, 0);
	meta = TrigpointPageGetMeta(mpage);
	if (!BufferIsValid(tbuf))
	{
		/* new tail page */
		Page		npage;

		nbuf = tp_new_page(index, MAIN_FORKNUM);	/* E3: a freed page when the FSM has one */
		npage = GenericXLogRegisterBuffer(state, nbuf, GENERIC_XLOG_FULL_IMAGE);
		tp_init_page(npage, BufferGetPageSize(nbuf), TP_PAGE_LOG, meta->generation);
		if (BlockNumberIsValid(meta->log_last))
		{
			Buffer		obuf = ReadBuffer(index, meta->log_last);
			Page		opage;

			LockBuffer(obuf, BUFFER_LOCK_EXCLUSIVE);
			opage = GenericXLogRegisterBuffer(state, obuf, 0);
			TrigpointPageGetOpaque(opage)->next = BufferGetBlockNumber(nbuf);
			tbuf = obuf;			/* released after GenericXLogFinish, which writes the copy back and dirties it */
		}
		else
			meta->log_first = BufferGetBlockNumber(nbuf);
		meta->log_last = BufferGetBlockNumber(nbuf);
		off = PageAddItem(npage, (Item) rec, size, InvalidOffsetNumber, false, false);
	}
	else
	{
		tpage = GenericXLogRegisterBuffer(state, tbuf, 0);
		off = PageAddItem(tpage, (Item) rec, size, InvalidOffsetNumber, false, false);
	}
	if (off == InvalidOffsetNumber)
	{
		GenericXLogAbort(state);
		if (BufferIsValid(nbuf)) UnlockReleaseBuffer(nbuf);
		if (BufferIsValid(tbuf)) UnlockReleaseBuffer(tbuf);
		UnlockReleaseBuffer(mbuf);
		elog(ERROR, "trigpoint: failed to add a log record to its page");
	}
	if (rec->kind == TP_REC_INSERT)
		meta->log_inserts++;
	else
		meta->log_deletes++;
	c->meta = *meta;			/* `meta` points into the xlog copy page, which Finish releases */
	GenericXLogFinish(state);
	/* apply locally and advance the cursor to our own record */
	tp_cache_apply_record(c, rec);
	c->log_blk = c->meta.log_last;
	c->log_off = off;
	if (BufferIsValid(nbuf))
		UnlockReleaseBuffer(nbuf);
	if (BufferIsValid(tbuf))
		UnlockReleaseBuffer(tbuf);
	UnlockReleaseBuffer(mbuf);
}

void
tp_cache_unpin(TrigpointCached *c)
{
	if (c == NULL)
		return;
	c->pins--;
	if (c->stale && c->pins <= 0)
		tp_cache_free(c);
}

/* ------------------------------------------------------------------------------------------------ */
/* scan */

/* One heap row this scan has looked at: the index's TID, the exact squared distance (INFINITY when the row was
 * dead, invisible to the snapshot or of the wrong dimension) and the visible version's TID. A row is fetched once
 * per scan however many batches re-present it. */
typedef struct TrigpointFetched
{
	uint64		key;			/* the index's TID (block << 16 | offset) */
	float		d2;
	ItemPointerData tid;
} TrigpointFetched;

typedef struct TrigpointScanOpaque
{
	TrigpointCached *cached;
	TupleTableSlot *slot;		/* heap fetches of candidates */
	MemoryContext tmpctx;		/* detoasted candidates, candidate lists: reset per batch */
	MemoryContext resctx;		/* the batch's heap and result list */
	AttrNumber	attno;			/* heap column of the indexed vector */
	bool		started;
	int			nres;
	int			cur;
	ItemPointerData *res_tid;
	float	   *res_d2;
	long long	stats[5];
	int			fetched;		/* heap rows read for exact distances, over the whole scan */
	int			dead;			/* candidates whose heap row was gone / invisible, over the whole scan */
	int			needed;			/* level-3 survivors with a bound below the batch's final tau: the floor for any exact phase */
	int			survivors;		/* level-3 survivors of the batch */
	/* Batches. The first batch is a top-trigpoint.k over trigpoint.probes cells, exactly
	 * the 0.4 computation. When the executor keeps asking (a selective WHERE above the index, a LIMIT above k), the
	 * next batch doubles k; under trigpoint.iterative_scan it also doubles the cells opened (to trigpoint.max_probes
	 * or every cell), keeps the core's routed cell order (tp_scan_restart), seeds tau from the rows already fetched
	 * and never fetches a heap row twice. A batch's rows are handed out in distance order, those handed out before
	 * skipped. The scan ends when a batch comes back short of its k with nothing left to open, when k covers the
	 * index, or at trigpoint.max_scan_tuples. */
	int			mode;			/* trigpoint.iterative_scan, read once at the first batch */
	int			batch;			/* batches computed so far */
	int			k_last;			/* k of the last batch */
	int			P_last;			/* cells asked of the last batch */
	int			P_max;			/* cells the iterative scan may open in all */
	bool		exhausted;		/* no further batch */
	bool		capped;			/* stopped by trigpoint.max_scan_tuples */
	void	   *sc;				/* the core's scan state, kept across batches (route once) */
	MemoryContext retctx;		/* the fetched rows and the hand-out set (live across batches) */
	HTAB	   *fetchtab;		/* key -> TrigpointFetched */
	HTAB	   *returned;		/* result TIDs handed out */
	float		last_d2;		/* strict_order: distance of the last row handed out */
	/* the batch's bounded max-heap over (d2, tid): the k closest fetched so far */
	int			hk;
	int			hn;
	float	   *hd2;
	ItemPointerData *htid;
	float	   *vbuf;			/* H11: a normalised copy of the fetched row for the cosine metric */
} TrigpointScanOpaque;

/* a failure reported by the core's scan: a value a page corruption put out of range is the index's fault (REINDEX),
 * anything else is ours */
static void
tp_core_scan_error(const char *msg)
{
	if (strncmp(msg, "corrupt image", 13) == 0)
		ereport(ERROR, (errcode(ERRCODE_INDEX_CORRUPTED), errmsg("trigpoint: %s", msg),
						errhint("Run trigpoint_check(<index>, true); REINDEX rebuilds the index.")));
	ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("trigpoint: %s", msg)));
}

static void
scan_core_end(TrigpointScanOpaque *so)
{
	if (so->sc)
	{
		tp_scan_end(so->sc);
		so->sc = NULL;
	}
}

IndexScanDesc
tp_beginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan = RelationGetIndexScan(index, nkeys, norderbys);
	TrigpointScanOpaque *so = (TrigpointScanOpaque *) palloc0(sizeof(TrigpointScanOpaque));

	so->cached = tp_cache_get(index);
	so->tmpctx = AllocSetContextCreate(CurrentMemoryContext, "trigpoint scan tmp", ALLOCSET_DEFAULT_SIZES);
	so->resctx = AllocSetContextCreate(CurrentMemoryContext, "trigpoint scan results", ALLOCSET_SMALL_SIZES);
	so->attno = index->rd_index->indkey.values[0];
	so->retctx = AllocSetContextCreate(CurrentMemoryContext, "trigpoint scan fetched", ALLOCSET_SMALL_SIZES);
	so->vbuf = (float *) palloc(sizeof(float) * (Size) Max(so->cached->D, 1));
	scan->opaque = so;
	return scan;
}

static void
returned_reset(TrigpointScanOpaque *so)
{
	HASHCTL		ctl;

	scan_core_end(so);
	if (so->returned)
		hash_destroy(so->returned);
	if (so->fetchtab)
		hash_destroy(so->fetchtab);
	MemoryContextReset(so->retctx);
	memset(&ctl, 0, sizeof(ctl));
	ctl.keysize = sizeof(uint64);
	ctl.entrysize = sizeof(uint64);
	ctl.hcxt = so->retctx;
	so->returned = hash_create("trigpoint returned tids", 256, &ctl, HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
	ctl.entrysize = sizeof(TrigpointFetched);
	so->fetchtab = hash_create("trigpoint fetched rows", 256, &ctl, HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
	so->batch = 0;
	so->k_last = 0;
	so->P_last = 0;
	so->fetched = so->dead = 0;
	so->exhausted = so->capped = false;
	so->last_d2 = -INFINITY;
}

void
tp_rescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys)
{
	TrigpointScanOpaque *so = (TrigpointScanOpaque *) scan->opaque;

	so->started = false;
	so->nres = so->cur = 0;
	MemoryContextReset(so->resctx);
	so->hd2 = NULL;
	so->htid = NULL;
	so->res_tid = NULL;
	so->res_d2 = NULL;
	returned_reset(so);
	if (keys && scan->numberOfKeys > 0)
		memmove(scan->keyData, keys, scan->numberOfKeys * sizeof(ScanKeyData));
	if (orderbys && scan->numberOfOrderBys > 0)
		memmove(scan->orderByData, orderbys, scan->numberOfOrderBys * sizeof(ScanKeyData));
}

static inline uint64
tid_key(const ItemPointerData *tid)
{
	return ((uint64) ItemPointerGetBlockNumber(tid) << 16) | ItemPointerGetOffsetNumber(tid);
}

/* the batch's max-heap: push (d2, tid), evicting the farthest once k are held; returns the new tau2 (k-th distance,
 * or INFINITY while fewer than k are held) */
static float
heap_push(TrigpointScanOpaque *so, float d2, const ItemPointerData *tid)
{
	int			i;

	if (so->hn < so->hk)
	{
		i = so->hn++;
		while (i > 0)
		{
			int			parent = (i - 1) / 2;

			if (so->hd2[parent] >= d2)
				break;
			so->hd2[i] = so->hd2[parent];
			so->htid[i] = so->htid[parent];
			i = parent;
		}
	}
	else
	{
		if (d2 >= so->hd2[0])
			return so->hd2[0];
		i = 0;
		for (;;)
		{
			int			l = 2 * i + 1, r = l + 1, big = i;
			float		bigd = d2;

			if (l < so->hn && so->hd2[l] > bigd)
			{
				big = l;
				bigd = so->hd2[l];
			}
			if (r < so->hn && so->hd2[r] > bigd)
				big = r;
			if (big == i)
				break;
			so->hd2[i] = so->hd2[big];
			so->htid[i] = so->htid[big];
			i = big;
		}
	}
	so->hd2[i] = d2;
	so->htid[i] = *tid;
	return so->hn == so->hk ? so->hd2[0] : INFINITY;
}

/* the heap's rows in ascending distance order into res_tid / res_d2 (the heap is consumed) */
static void
heap_to_result(TrigpointScanOpaque *so)
{
	int			n = so->hn;

	so->res_tid = (ItemPointerData *) MemoryContextAlloc(so->resctx, sizeof(ItemPointerData) * Max(n, 1));
	so->res_d2 = (float *) MemoryContextAlloc(so->resctx, sizeof(float) * Max(n, 1));
	so->nres = n;
	while (so->hn > 0)
	{
		float		d2 = so->hd2[0];
		ItemPointerData tid = so->htid[0];
		float		last = so->hd2[--so->hn];
		ItemPointerData lasttid = so->htid[so->hn];
		int			i = 0;

		so->res_d2[so->hn] = d2;
		so->res_tid[so->hn] = tid;
		if (so->hn == 0)
			break;
		for (;;)					/* sift the last element down from the root */
		{
			int			l = 2 * i + 1, r = l + 1, big = i;
			float		bigd = last;

			if (l < so->hn && so->hd2[l] > bigd)
			{
				big = l;
				bigd = so->hd2[l];
			}
			if (r < so->hn && so->hd2[r] > bigd)
				big = r;
			if (big == i)
				break;
			so->hd2[i] = so->hd2[big];
			so->htid[i] = so->htid[big];
			i = big;
		}
		so->hd2[i] = last;
		so->htid[i] = lasttid;
	}
	so->cur = 0;
}

/* one batch: the top-k over the first P cells of the routed order, tau seeded from the rows fetched by earlier batches */
static void
tp_compute(IndexScanDesc scan)
{
	TrigpointScanOpaque *so = (TrigpointScanOpaque *) scan->opaque;
	TrigpointCached *c = so->cached;
	ScanKey		sk;
	Vector	   *q;
	MemoryContext old;
	float		ch[4] = {1, 1, 1, 1};
	int			k, M = tp_hits, P, ef = tp_ef > 0 ? tp_ef : tp_hits;
	int			nP;
	float		tau2 = INFINITY;
	const float slack = 1.0f + 1e-5f;
	BlockNumber heap_nblocks;
	bool		iterative;
	instr_time	t_start, t_now, t_core, t_fetch, t_route;

	INSTR_TIME_SET_ZERO(t_route);
	INSTR_TIME_SET_ZERO(t_core);
	INSTR_TIME_SET_ZERO(t_fetch);
	INSTR_TIME_SET_CURRENT(t_start);
	so->started = true;
	so->nres = so->cur = 0;
	so->survivors = so->needed = 0;
	memset(so->stats, 0, sizeof(so->stats));
	if (so->batch == 0)
	{
		so->mode = tp_iterative;
		so->P_max = (tp_max_probes > 0 && tp_max_probes < c->C) ? tp_max_probes : c->C;
		k = tp_k;
		P = tp_exact ? c->C : tp_probes;
	}
	else
	{
		k = (so->k_last > c->n / 2) ? c->n : so->k_last * 2;
		if (tp_exact)
			P = c->C;
		else if (so->mode == TP_ITER_OFF)
			P = so->P_last;
		else
			P = Min(so->P_last * 2, Max(so->P_max, so->P_last));
	}
	iterative = so->mode != TP_ITER_OFF;
	if (k < 1)
		k = 1;
	so->k_last = k;
	so->P_last = P;
	so->batch++;
	if (c->n == 0)
	{
		so->exhausted = true;
		return;
	}
	if (scan->numberOfOrderBys != 1)
		ereport(ERROR, (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
						errmsg("trigpoint: index scans need exactly one ORDER BY <column> <-> <query>")));
	if (scan->numberOfKeys > 0)
		ereport(ERROR, (errcode(ERRCODE_FEATURE_NOT_SUPPORTED), errmsg("trigpoint: index quals are not supported")));
	sk = &scan->orderByData[0];
	if (sk->sk_flags & SK_ISNULL)
	{
		so->exhausted = true;
		return;
	}

	MemoryContextReset(so->tmpctx);
	old = MemoryContextSwitchTo(so->tmpctx);
	q = DatumGetVector(sk->sk_argument);
	if (q->dim != c->D)
		ereport(ERROR, (errcode(ERRCODE_DATA_EXCEPTION),
						errmsg("trigpoint: query has %d dimensions, index has %d", q->dim, c->D)));
	if (c->meta.metric == TP_METRIC_COSINE)
	{
		/* a private copy: the datum may be the caller's; the index ranks by L2 on the unit sphere */
		Vector	   *qn = (Vector *) palloc(VARSIZE_ANY(q));

		memcpy(qn, q, VARSIZE_ANY(q));
		tp_normalize(qn->x, qn->dim);
		q = qn;
	}
	if (so->slot == NULL)
	{
		MemoryContextSwitchTo(old);
		so->slot = table_slot_create(scan->heapRelation, NULL);
		MemoryContextSwitchTo(so->tmpctx);
	}
	MemoryContextReset(so->resctx);
	so->hk = k;
	so->hn = 0;
	so->hd2 = (float *) MemoryContextAlloc(so->resctx, sizeof(float) * k);
	so->htid = (ItemPointerData *) MemoryContextAlloc(so->resctx, sizeof(ItemPointerData) * k);
	so->res_tid = NULL;
	so->res_d2 = NULL;
	heap_nblocks = RelationGetNumberOfBlocks(scan->heapRelation);

	if (tp_exact)
		;						/* certificate: exact bound (ch = 1) over every cell */
	else
		tp_index_ch(c->ix, tp_delta, ch);

	INSTR_TIME_SET_CURRENT(t_now);
	if (so->sc == NULL)
	{
		if (tp_debug_no_delta_graph && c->dgraph)
			tp_index_set_delta_graph(c->ix, NULL);	/* diagnostic: the router ignores the appended positions for this scan */
		so->sc = tp_scan_begin(c->ix, q->x, M, P, ef, ch);
		if (tp_debug_no_delta_graph && c->dgraph)
			tp_index_set_delta_graph(c->ix, c->dgraph);	/* the route is done; the cells are scanned without the router */
		if (so->sc == NULL)
			tp_core_scan_error(tp_last_error());
		nP = Min(P, c->C);
	}
	else
	{
		/* a later batch: the routed order is kept, the cursor rewound; the opened cells are cascaded again under the
		 * larger tau, the rows fetched before answered from the table */
		nP = tp_scan_restart(so->sc, P);
		if (nP < 0)
			tp_core_scan_error(tp_last_error());
	}
	PG_TRY();
	{
		int			m;
		int			step = 0;
		int			chunk = Max(tp_seed_cells, 32);
		int			nlb = 0, caplb = 1024;
		float	   *all_lb = (float *) palloc(sizeof(float) * caplb);	/* every survivor's bound, for "needed" */
		instr_time	t_a, t_b;

		if (so->batch > 1)
		{
			/* seed the heap, and so tau, from the rows fetched by the earlier batches */
			HASH_SEQ_STATUS seq;
			TrigpointFetched *f;

			hash_seq_init(&seq, so->fetchtab);
			while ((f = (TrigpointFetched *) hash_seq_search(&seq)) != NULL)
				if (f->d2 < INFINITY)
					tau2 = heap_push(so, f->d2, &f->tid);
		}
		INSTR_TIME_SET_CURRENT(t_a);
		INSTR_TIME_ACCUM_DIFF(t_route, t_a, t_now);	/* route: project + beam + rank (the first batch only) */
		for (;;)
		{
			int		   *ids;
			float	   *lb2;
			int			t;

			if (iterative && so->fetched >= tp_max_scan_tuples)
			{
				so->capped = true;
				break;
			}
			INSTR_TIME_SET_CURRENT(t_a);
			/* cell by cell while seeding tau, then (merge mode) every remaining cell at once. A later batch's k is
			 * double the last one's, so its seeded heap is not full: it keeps going cell by cell until tau is finite,
			 * or the merged rest would cascade every cell unpruned. */
			if (tp_merge && step >= tp_seed_cells && so->batch == 1)
				m = tp_scan_rest(so->sc, tau2);
			else if (tp_merge && step >= tp_seed_cells && tau2 < INFINITY)
			{
				/* a later batch takes the rest in growing chunks: the fetches of one chunk tighten the tau of the next */
				m = tp_scan_some(so->sc, tau2, chunk);
				chunk = Min(chunk * 2, c->C);
			}
			else
				m = tp_scan_cell(so->sc, tau2);
			INSTR_TIME_SET_CURRENT(t_b);
			INSTR_TIME_ACCUM_DIFF(t_core, t_b, t_a);
			step++;
			if (m == -1)
				break;
			if (m == -2)
				tp_core_scan_error(tp_last_error());
			if (m == 0)
				continue;
			ids = (int *) palloc(sizeof(int) * m);
			lb2 = (float *) palloc(sizeof(float) * m);
			tp_scan_take(so->sc, ids, lb2);
			if (nlb + m > caplb)
			{
				while (nlb + m > caplb)
					caplb *= 2;
				all_lb = (float *) repalloc(all_lb, sizeof(float) * caplb);
			}
			memcpy(all_lb + nlb, lb2, sizeof(float) * m);
			nlb += m;
			for (t = 0; t < m; t++)
			{
				ItemPointerData tid;
				bool		call_again = false, all_dead = false, isnull, found;
				Datum		d;
				Vector	   *v;
				TrigpointFetched *f;
				uint64		key;

				if (lb2[t] > tau2 * slack)
					break;		/* the rest of this cell is bounded away (sorted by bound) */
				if (iterative && so->fetched >= tp_max_scan_tuples)
				{
					so->capped = true;
					break;
				}
				if (!tp_tid_of(c, ids[t], &tid))
				{
					so->dead++;	/* tombstoned (the core should not have returned it) or out of range */
					continue;
				}
				key = tid_key(&tid);
				f = (TrigpointFetched *) hash_search(so->fetchtab, &key, HASH_ENTER, &found);
				if (found)
					continue;	/* fetched by an earlier batch: in the heap already, or beyond its seeded tau */
				f->d2 = INFINITY;
				f->tid = tid;
				if (tp_debug_skip_heap)
				{
					/* diagnostic only: rank by lower bound, touch no heap page (isolates the core's time) */
					f->d2 = lb2[t];
					tau2 = heap_push(so, lb2[t], &tid);
					continue;
				}
				if (ItemPointerGetBlockNumber(&tid) >= heap_nblocks)
				{
					so->dead++;	/* heap truncated by VACUUM behind this TID */
					continue;
				}
				if (!table_index_fetch_tuple(scan->xs_heapfetch, &tid, scan->xs_snapshot, so->slot, &call_again, &all_dead))
				{
					so->dead++;	/* deleted / not visible to this snapshot */
					continue;
				}
				so->fetched++;
				d = slot_getattr(so->slot, so->attno, &isnull);
				if (!isnull)
				{
					v = DatumGetVector(d);
					if (v->dim == c->D)
					{
						if (c->meta.metric == TP_METRIC_COSINE)
						{
							/* the slot's datum may point into the buffer: rank a normalised copy */
							memcpy(so->vbuf, v->x, sizeof(float) * (Size) c->D);
							tp_normalize(so->vbuf, c->D);
							f->d2 = tp_sq_dist(q->x, so->vbuf, c->D);
						}
						else
							f->d2 = tp_sq_dist(q->x, v->x, c->D);
						f->tid = so->slot->tts_tid;
						tau2 = heap_push(so, f->d2, &f->tid);
					}
				}
				ExecClearTuple(so->slot);
			}
			pfree(ids);
			pfree(lb2);
			INSTR_TIME_SET_CURRENT(t_a);
			INSTR_TIME_ACCUM_DIFF(t_fetch, t_a, t_b);	/* heap fetches + exact distances of this cell */
			CHECK_FOR_INTERRUPTS();
			if (so->capped)
				break;
		}
		tp_scan_stats(so->sc, so->stats);
		nP = (int) so->stats[4];	/* cells the order covers (fewer than P when the index has fewer non-empty cells) */
		so->survivors = nlb;
		for (m = 0; m < nlb; m++)
			if (all_lb[m] < tau2)
				so->needed++;
	}
	PG_CATCH();
	{
		scan_core_end(so);
		PG_RE_THROW();
	}
	PG_END_TRY();
	heap_to_result(so);
	/* the next batch, if the executor asks: unless the index is covered (k >= n), the batch came back short of its k
	 * with no more cells to open, or the tuple cap stopped it */
	{
		bool		all_cells = (nP < P) || (nP >= c->C);
		bool		can_widen = iterative && !tp_exact && so->P_last < so->P_max && !all_cells;

		if (so->capped || k >= c->n || (so->nres < k && !can_widen))
			so->exhausted = true;
	}
	if (so->exhausted)
		scan_core_end(so);
	MemoryContextSwitchTo(old);
	MemoryContextReset(so->tmpctx);
	if (tp_debug)
	{
		INSTR_TIME_SET_CURRENT(t_now);
		ereport(NOTICE, (errmsg("trigpoint: %s batch=%d%s%s iterative=%s k=%d M=%d P=%d/%d%s%s delta=%.3g ch=%.3f,%.3f,%.3f cells_hit=%lld scanned=%lld dim_ops=%lld "
								"survivors=%d needed=%d heap_fetched=%d dead=%d total=%.3f ms core=%.3f ms (route %.3f) heap=%.3f ms"
								" router_mode=%d n=%d n_base=%d delta_graph=%d/%d%s",
								c->seg ? "shared" : "owned", so->batch, so->capped ? " (capped)" : "", so->exhausted ? " (last)" : "",
								so->mode == TP_ITER_OFF ? "off" : so->mode == TP_ITER_STRICT ? "strict_order" : "relaxed_order",
								k, M, so->stats[3] < nP ? (int) so->stats[3] : nP, c->C, tp_exact ? " exact" : "",
								tp_merge ? " merged" : "", tp_delta,
								ch[0], ch[1], ch[2], so->stats[2], so->stats[3], so->stats[0], so->survivors, so->needed, so->fetched, so->dead,
								INSTR_TIME_GET_MILLISEC(t_now) - INSTR_TIME_GET_MILLISEC(t_start),
								INSTR_TIME_GET_MILLISEC(t_core) + INSTR_TIME_GET_MILLISEC(t_route), INSTR_TIME_GET_MILLISEC(t_route),
								INSTR_TIME_GET_MILLISEC(t_fetch), tp_index_router_mode(c->ix), c->n, c->n_base,
								c->dgraph ? ((FlatGraph *) c->dgraph)->n : 0, c->dgraph ? ((FlatGraph *) c->dgraph)->n_cap : 0,
								tp_debug_no_delta_graph ? " (delta graph detached for the route)" : "")));
	}
}

bool
tp_gettuple(IndexScanDesc scan, ScanDirection dir)
{
	TrigpointScanOpaque *so = (TrigpointScanOpaque *) scan->opaque;

	if (dir != ForwardScanDirection)
		ereport(ERROR, (errcode(ERRCODE_FEATURE_NOT_SUPPORTED), errmsg("trigpoint: backward scans are not supported")));
	if (so->returned == NULL)
		returned_reset(so);
	for (;;)
	{
		if (!so->started)
			tp_compute(scan);
		while (so->cur < so->nres)
		{
			ItemPointerData tid = so->res_tid[so->cur];
			float		d2 = so->res_d2[so->cur];
			uint64		key = tid_key(&tid);
			bool		found;

			so->cur++;
			hash_search(so->returned, &key, HASH_ENTER, &found);
			if (found)
				continue;		/* handed out by an earlier, narrower batch */
			if (so->mode == TP_ITER_STRICT && d2 < so->last_d2)
				continue;		/* strict_order: a later batch found it, but the stream has passed its distance */
			so->last_d2 = d2;
			scan->xs_heaptid = tid;
			scan->xs_recheck = false;
			scan->xs_recheckorderby = false;
			return true;
		}
		if (so->exhausted)
			return false;
		so->started = false;
		CHECK_FOR_INTERRUPTS();
	}
}

void
tp_endscan(IndexScanDesc scan)
{
	TrigpointScanOpaque *so = (TrigpointScanOpaque *) scan->opaque;

	scan_core_end(so);
	if (so->slot)
		ExecDropSingleTupleTableSlot(so->slot);
	if (so->returned)
		hash_destroy(so->returned);
	if (so->fetchtab)
		hash_destroy(so->fetchtab);
	MemoryContextDelete(so->retctx);
	MemoryContextDelete(so->tmpctx);
	MemoryContextDelete(so->resctx);
	tp_cache_unpin(so->cached);
	pfree(so);
	scan->opaque = NULL;
}

/* ------------------------------------------------------------------------------------------------ */
/* planner */

void
tp_costestimate(PlannerInfo *root, IndexPath *path, double loop_count, Cost *indexStartupCost,
				Cost *indexTotalCost, Selectivity *indexSelectivity, double *indexCorrelation, double *indexPages)
{
	GenericCosts costs;

	/* only useful for ORDER BY <-> ; never for a plain scan */
	if (path->indexorderbys == NIL)
	{
		*indexStartupCost = DBL_MAX;
		*indexTotalCost = DBL_MAX;
		*indexSelectivity = 0;
		*indexCorrelation = 0;
		*indexPages = 0;
		return;
	}
	MemSet(&costs, 0, sizeof(costs));
	/* At most trigpoint.k rows come back; the image is searched from the backend cache, so the
	 * index pages do not scale the cost (pgvector's estimates likewise make the index win over a
	 * sort whenever the query has ORDER BY <->). The core's work per query — the beam over
	 * trigpoint.hits router nodes and P cells of ~75 sketches — is charged as operator evaluations. */
	costs.numIndexTuples = tp_k;
	genericcostestimate(root, path, loop_count, &costs);
	costs.indexTotalCost += ((double) tp_hits * 8.0 + (double) tp_probes * 75.0) * cpu_operator_cost;
	*indexStartupCost = costs.indexTotalCost;	/* the whole top-k is computed before the first row */
	*indexTotalCost = costs.indexTotalCost;
	*indexSelectivity = costs.indexSelectivity;
	*indexCorrelation = costs.indexCorrelation;
	*indexPages = costs.numIndexPages;
}
