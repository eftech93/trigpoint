/*
 * tp_check.c — trigpoint_check(regclass, deep): the consistency checker, and the
 * test-only page peek / poke used by the corruption tests.
 *
 * The checker reads the index's pages only (never the shared segment): the meta page, the image chain
 * of the current generation (every page's kind, generation and item size, the chain's total against
 * the meta page), every log page and record (through tp_log_record, the same check the replay runs),
 * the TID table against the heap's block count, and the flat image through tpfl_check (the same cheap
 * check every open runs; deep = 1 also walks the arrays and the router graph). Every other page is
 * classified (retired image / log page of another generation awaiting the FSM, new, unknown) and
 * counted. Findings come back one per row; the last row is the summary with the problem count, so a
 * caller can `SELECT ... WHERE line LIKE 'summary:%'`.
 */
#include "trigpoint.h"

#include <math.h>

#include "access/generic_xlog.h"
#include "access/htup_details.h"
#include "access/relation.h"
#include "catalog/index.h"
#include "commands/defrem.h"
#include "funcapi.h"
#include "lib/stringinfo.h"
#include "miscadmin.h"
#include "nodes/pg_list.h"
#include "storage/bufmgr.h"
#include "utils/builtins.h"
#include "utils/memutils.h"
#include "utils/rel.h"

#include "flat_check.h"

PG_FUNCTION_INFO_V1(trigpoint_check);
#ifdef TRIGPOINT_TEST_HOOKS
PG_FUNCTION_INFO_V1(trigpoint_debug_peek);
PG_FUNCTION_INFO_V1(trigpoint_debug_poke);
#endif

typedef struct CheckState
{
	List	   *lines;			/* text lines, in order */
	int			problems;
} CheckState;

static void
note(CheckState *st, bool problem, const char *fmt,...) pg_attribute_printf(3, 4);

static void
note(CheckState *st, bool problem, const char *fmt,...)
{
	StringInfoData s;
	va_list		ap;

	initStringInfo(&s);
	appendStringInfoString(&s, problem ? "problem: " : "ok: ");
	for (;;)
	{
		int			need;

		va_start(ap, fmt);
		need = appendStringInfoVA(&s, fmt, ap);
		va_end(ap);
		if (need == 0)
			break;
		enlargeStringInfo(&s, need);
	}
	st->lines = lappend(st->lines, s.data);
	if (problem)
		st->problems++;
}

/* the page's special space as ours, or NULL when the page is new or its special size is not ours */
static TrigpointPageOpaque
page_opaque(Page page)
{
	if (PageIsNew(page) || PageGetSpecialSize(page) != MAXALIGN(sizeof(TrigpointPageOpaqueData)))
		return NULL;
	return TrigpointPageGetOpaque(page);
}

/* the image chain of the current generation; returns the blob (palloc'd) when it is whole, else NULL */
static char *
check_image_chain(CheckState *st, Relation index, const TrigpointMetaPageData *meta, BlockNumber nblocks, uint8 *seen)
{
	BlockNumber blk = meta->image_start;
	uint64		off = 0;
	uint32		i;
	char	   *blob;
	bool		whole = true;

	if (meta->blob_bytes > (uint64) MaxAllocHugeSize)
	{
		note(st, true, "image of " UINT64_FORMAT " bytes is larger than this backend can hold; chain not read", meta->blob_bytes);
		return NULL;
	}
	blob = (char *) palloc_extended((Size) meta->blob_bytes, MCXT_ALLOC_HUGE | MCXT_ALLOC_NO_OOM);
	if (blob == NULL)
	{
		note(st, true, "out of memory for the " UINT64_FORMAT "-byte image; chain not read", meta->blob_bytes);
		return NULL;
	}
	for (i = 0; i < meta->nchunks; i++)
	{
		Buffer		buf;
		Page		page;
		TrigpointPageOpaque op;
		ItemId		iid;
		Size		sz;

		if (!BlockNumberIsValid(blk) || blk >= nblocks)
		{
			note(st, true, "image chain ends after %u of %u pages (next = %u, relation has %u blocks)", i, meta->nchunks, blk, nblocks);
			whole = false;
			break;
		}
		if (seen[blk])
		{
			note(st, true, "image chain revisits block %u after %u pages (cycle)", blk, i);
			whole = false;
			break;
		}
		seen[blk] = 1;
		buf = ReadBuffer(index, blk);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		op = page_opaque(page);
		if (op == NULL || op->kind != TP_PAGE_IMAGE || op->generation != meta->generation)
		{
			note(st, true, "block %u in the image chain is not an image page of generation %u (kind %u, generation %u)",
				 blk, meta->generation, op ? op->kind : 0, op ? op->generation : 0);
			UnlockReleaseBuffer(buf);
			whole = false;
			break;
		}
		if (PageGetMaxOffsetNumber(page) != FirstOffsetNumber)
		{
			note(st, true, "image page %u holds %u items, not one", blk, PageGetMaxOffsetNumber(page));
			UnlockReleaseBuffer(buf);
			whole = false;
			break;
		}
		iid = PageGetItemId(page, FirstOffsetNumber);
		sz = ItemIdGetLength(iid);
		if (!ItemIdIsNormal(iid) || sz == 0 || sz > TP_CHUNK_BYTES || off + sz > meta->blob_bytes)
		{
			note(st, true, "image page %u item of %zu bytes at image offset " UINT64_FORMAT " does not fit the " UINT64_FORMAT "-byte image",
				 blk, sz, off, meta->blob_bytes);
			UnlockReleaseBuffer(buf);
			whole = false;
			break;
		}
		memcpy(blob + off, PageGetItem(page, iid), sz);
		off += sz;
		blk = op->next;
		UnlockReleaseBuffer(buf);
		CHECK_FOR_INTERRUPTS();
	}
	if (whole && off != meta->blob_bytes)
	{
		note(st, true, "image chain holds " UINT64_FORMAT " bytes, the meta page says " UINT64_FORMAT, off, meta->blob_bytes);
		whole = false;
	}
	if (whole && BlockNumberIsValid(blk))
		note(st, true, "image chain continues past its %u pages (next = %u)", meta->nchunks, blk);
	if (whole)
		note(st, false, "image chain: %u pages, " UINT64_FORMAT " bytes", meta->nchunks, off);
	if (!whole)
	{
		pfree(blob);
		return NULL;
	}
	return blob;
}

/* the log chain: every record checked as the replay checks it, ids in sequence, tombstones in range */
static void
check_log_chain(CheckState *st, Relation index, const TrigpointMetaPageData *meta, BlockNumber nblocks, uint8 *seen)
{
	BlockNumber blk = meta->log_first;
	int64		inserts = 0,
				deletes = 0,
				pages = 0;
	int32		next_id = meta->n_image;
	bool		whole = true;

	if (!BlockNumberIsValid(blk))
	{
		if (BlockNumberIsValid(meta->log_last) || meta->log_inserts != 0 || meta->log_deletes != 0)
			note(st, true, "no log chain but log_last = %u, log_inserts = " INT64_FORMAT ", log_deletes = " INT64_FORMAT,
				 meta->log_last, meta->log_inserts, meta->log_deletes);
		else
			note(st, false, "log: empty");
		return;
	}
	for (;;)
	{
		Buffer		buf;
		Page		page;
		TrigpointPageOpaque op;
		OffsetNumber max,
					off;
		BlockNumber next;

		if (blk >= nblocks)
		{
			note(st, true, "log chain points at block %u beyond the relation's %u blocks", blk, nblocks);
			whole = false;
			break;
		}
		if (seen[blk])
		{
			note(st, true, "log chain revisits block %u (cycle, or an image page)", blk);
			whole = false;
			break;
		}
		seen[blk] = 1;
		buf = ReadBuffer(index, blk);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		op = page_opaque(page);
		if (op == NULL || op->kind != TP_PAGE_LOG || op->generation != meta->generation)
		{
			note(st, true, "block %u in the log chain is not a log page of generation %u (kind %u, generation %u)",
				 blk, meta->generation, op ? op->kind : 0, op ? op->generation : 0);
			UnlockReleaseBuffer(buf);
			whole = false;
			break;
		}
		pages++;
		max = PageGetMaxOffsetNumber(page);
		for (off = FirstOffsetNumber; off <= max; off++)
		{
			ItemId		iid = PageGetItemId(page, off);
			TrigpointLogRecord *rec;
			char	   *why;

			if (!ItemIdIsUsed(iid))
				continue;
			/* never catch an ERROR under a buffer lock: errfinish resets the interrupt hold-off count and the
			 * release would underflow it (found by the H3 tests: the backend could not be terminated afterwards) */
			rec = tp_log_record_problem(page, off, blk, meta->L, meta->lv, meta->C, &why);
			if (rec == NULL)
			{
				note(st, true, "%s", why);
				whole = false;
				break;
			}
			if (rec->kind == TP_REC_INSERT)
			{
				if (rec->id != next_id)
				{
					note(st, true, "log record at block %u offset %u has id %d, expected %d", blk, off, rec->id, next_id);
					whole = false;
					break;
				}
				next_id++;
				inserts++;
			}
			else
			{
				if (rec->id >= next_id)
				{
					note(st, true, "tombstone at block %u offset %u names id %d but only %d ids exist", blk, off, rec->id, next_id);
					whole = false;
					break;
				}
				deletes++;
			}
		}
		next = op->next;
		UnlockReleaseBuffer(buf);
		if (!whole)
			break;
		if (!BlockNumberIsValid(next))
		{
			if (blk != meta->log_last)
				note(st, true, "log chain ends at block %u but the meta page says log_last = %u", blk, meta->log_last);
			break;
		}
		blk = next;
		CHECK_FOR_INTERRUPTS();
	}
	if (whole)
	{
		if (inserts != meta->log_inserts || deletes != meta->log_deletes)
			note(st, true, "log holds " INT64_FORMAT " inserts and " INT64_FORMAT " tombstones, the meta page counts " INT64_FORMAT " and " INT64_FORMAT,
				 inserts, deletes, meta->log_inserts, meta->log_deletes);
		note(st, false, "log: " INT64_FORMAT " pages, " INT64_FORMAT " inserts, " INT64_FORMAT " tombstones, ids %d..%d",
			 pages, inserts, deletes, meta->n_image, next_id - 1);
	}
}

Datum
trigpoint_check(PG_FUNCTION_ARGS)
{
	FuncCallContext *fctx;

	if (SRF_IS_FIRSTCALL())
	{
		Oid			relid = PG_GETARG_OID(0);
		bool		deep = PG_GETARG_BOOL(1);
		MemoryContext old;
		CheckState *st;
		Relation	index;
		TrigpointMetaPageData meta;
		BlockNumber nblocks,
					blk;
		uint8	   *seen;
		char	   *blob = NULL;
		int64		n_new = 0,
					n_retired_image = 0,
					n_retired_log = 0,
					n_unknown = 0,
					n_wrong_size = 0;
		Buffer		mbuf;
		Page		mpage;
		bool		meta_ok = true;

		fctx = SRF_FIRSTCALL_INIT();
		old = MemoryContextSwitchTo(fctx->multi_call_memory_ctx);
		memset(&meta, 0, sizeof meta);
		st = (CheckState *) palloc0(sizeof(CheckState));
		fctx->user_fctx = st;

		index = tp_open_index_checked(relid, AccessShareLock, deep);	/* deep takes the index apart: owner or superuser */
		nblocks = RelationGetNumberOfBlocks(index);
		if (nblocks == 0)
		{
			note(st, true, "relation has no pages");
			meta_ok = false;
		}
		else
		{
			mbuf = ReadBuffer(index, TP_METAPAGE_BLKNO);
			LockBuffer(mbuf, BUFFER_LOCK_SHARE);
			mpage = BufferGetPage(mbuf);
			if (PageIsNew(mpage) || PageGetSpecialSize(mpage) != 0 ||
				((PageHeader) mpage)->pd_lower < SizeOfPageHeaderData + sizeof(TrigpointMetaPageData))
			{
				note(st, true, "block 0 is not a meta page");
				meta_ok = false;
			}
			else
				memcpy(&meta, TrigpointPageGetMeta(mpage), sizeof meta);
			UnlockReleaseBuffer(mbuf);
		}
		if (meta_ok)
		{
			char	   *why = tp_meta_problem(&meta, nblocks);	/* the same check every reader runs */

			if (why != NULL)
			{
				note(st, true, "meta page: %s", why);
				meta_ok = false;
			}
			if (meta_ok)
				note(st, false, "meta: n=%d n_image=%d D=%d cells=%d levels=%d generation=%u image_pages=%u rel_pages=%u",
					 meta.n, meta.n_image, meta.D, meta.C, meta.L, meta.generation, meta.nchunks, nblocks);
		}
		seen = (uint8 *) palloc0(nblocks ? nblocks : 1);
		if (nblocks)
			seen[0] = 1;
		if (meta_ok && meta.core_bytes > 0)
		{
			blob = check_image_chain(st, index, &meta, nblocks, seen);
			check_log_chain(st, index, &meta, nblocks, seen);
		}
		else if (meta_ok)
			note(st, false, "model-less index (no image; written before 0.3)");
		/* every other page */
		for (blk = 0; blk < nblocks; blk++)
		{
			Buffer		buf;
			Page		page;
			TrigpointPageOpaque op;

			if (seen[blk])
				continue;
			buf = ReadBuffer(index, blk);
			LockBuffer(buf, BUFFER_LOCK_SHARE);
			page = BufferGetPage(buf);
			if (PageIsNew(page))
				n_new++;
			else if ((op = page_opaque(page)) == NULL)
				n_wrong_size++;
			else if (op->kind == TP_PAGE_IMAGE)
				n_retired_image++;
			else if (op->kind == TP_PAGE_LOG)
				n_retired_log++;
			else
				n_unknown++;
			UnlockReleaseBuffer(buf);
			if ((blk & 1023) == 0)
				CHECK_FOR_INTERRUPTS();
		}
		if (n_wrong_size || n_unknown)
			note(st, true, "pages outside every chain: " INT64_FORMAT " with a foreign special size, " INT64_FORMAT " of unknown kind", n_wrong_size, n_unknown);
		note(st, false, "pages outside the current chains: " INT64_FORMAT " retired image, " INT64_FORMAT " retired log, " INT64_FORMAT " new",
			 n_retired_image, n_retired_log, n_new);
		/* the TID table against the heap, then the flat image */
		if (blob != NULL)
		{
			Relation	heap = relation_open(index->rd_index->indrelid, AccessShareLock);
			BlockNumber hblocks = RelationGetNumberOfBlocks(heap);
			const uint64 *tids = (const uint64 *) blob;
			int64		bad = 0,
						dead = 0;
			int32		i;

			for (i = 0; i < meta.n_image; i++)
			{
				if (tids[i] == 0)
					dead++;
				else if ((tids[i] >> 16) >= hblocks || (tids[i] & 0xFFFF) == 0 || (tids[i] & 0xFFFF) > MaxHeapTuplesPerPage)
					bad++;
			}
			relation_close(heap, AccessShareLock);
			if (bad)
				note(st, true, "TID table: " INT64_FORMAT " of %d entries point outside the heap (%u blocks)", bad, meta.n_image, hblocks);
			else
				note(st, false, "TID table: %d entries, " INT64_FORMAT " tombstoned at image time, heap has %u blocks", meta.n_image, dead, hblocks);
			if (tp_flat_check(blob + (Size) meta.n_image * sizeof(uint64), (size_t) meta.core_bytes, deep ? 1 : 0) < 0)
				note(st, true, "%s", tp_last_error());
			else
			{
				const FlatHeader *h = (const FlatHeader *) (blob + (Size) meta.n_image * sizeof(uint64));

				if (h->n != meta.n_image || h->D != meta.D || h->C != meta.C || h->L != meta.L)
					note(st, true, "flat image header (n=%d D=%d C=%d L=%d) disagrees with the meta page (n_image=%d D=%d C=%d L=%d)",
						 h->n, h->D, h->C, h->L, meta.n_image, meta.D, meta.C, meta.L);
				else
					note(st, false, "flat image: %s check passed (" UINT64_FORMAT " bytes, graph " UINT64_FORMAT " bytes)",
						 deep ? "deep" : "header and section", h->total_bytes, h->graph_bytes);
			}
			pfree(blob);
		}
		relation_close(index, AccessShareLock);
		{
			StringInfoData s;

			initStringInfo(&s);
			appendStringInfo(&s, "summary: %s, %d problems, %u pages, %s", st->problems ? "CORRUPT" : "ok", st->problems, nblocks, deep ? "deep" : "shallow");
			st->lines = lappend(st->lines, s.data);
		}
		fctx->max_calls = list_length(st->lines);
		MemoryContextSwitchTo(old);
	}
	fctx = SRF_PERCALL_SETUP();
	if (fctx->call_cntr < fctx->max_calls)
	{
		CheckState *st = (CheckState *) fctx->user_fctx;
		char	   *line = (char *) list_nth(st->lines, (int) fctx->call_cntr);

		SRF_RETURN_NEXT(fctx, PointerGetDatum(cstring_to_text(line)));
	}
	SRF_RETURN_DONE(fctx);
}

/* ---- test hooks (test builds only: make TEST_HOOKS=1 / -DTRIGPOINT_TEST_HOOKS; absent from release builds) ---- */
#ifdef TRIGPOINT_TEST_HOOKS

static Relation
debug_open(Oid relid, int blkno, int offset, Buffer *buf)
{
	Relation	index;

	if (!superuser())
		ereport(ERROR, (errcode(ERRCODE_INSUFFICIENT_PRIVILEGE), errmsg("trigpoint: the page hooks are superuser-only")));
	if (!tp_debug)
		ereport(ERROR, (errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE), errmsg("trigpoint: the page hooks need trigpoint.debug = on")));
	index = relation_open(relid, RowExclusiveLock);
	if (index->rd_rel->relkind != RELKIND_INDEX || index->rd_rel->relam != get_index_am_oid("trigpoint", false))
		ereport(ERROR, (errcode(ERRCODE_WRONG_OBJECT_TYPE), errmsg("\"%s\" is not a trigpoint index", RelationGetRelationName(index))));
	if (blkno < 0 || (BlockNumber) blkno >= RelationGetNumberOfBlocks(index))
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("trigpoint: block %d is outside the index", blkno)));
	if (offset < 0 || offset >= BLCKSZ)
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("trigpoint: offset %d is outside the page", offset)));
	*buf = ReadBuffer(index, (BlockNumber) blkno);
	return index;
}

Datum
trigpoint_debug_peek(PG_FUNCTION_ARGS)
{
	int			blkno = PG_GETARG_INT32(1),
				offset = PG_GETARG_INT32(2);
	Buffer		buf;
	Relation	index = debug_open(PG_GETARG_OID(0), blkno, offset, &buf);
	int			v;

	LockBuffer(buf, BUFFER_LOCK_SHARE);
	v = ((unsigned char *) BufferGetPage(buf))[offset];
	UnlockReleaseBuffer(buf);
	relation_close(index, RowExclusiveLock);
	PG_RETURN_INT32(v);
}

Datum
trigpoint_debug_poke(PG_FUNCTION_ARGS)
{
	int			blkno = PG_GETARG_INT32(1),
				offset = PG_GETARG_INT32(2),
				value = PG_GETARG_INT32(3);
	Buffer		buf;
	Relation	index = debug_open(PG_GETARG_OID(0), blkno, offset, &buf);
	GenericXLogState *state;
	Page		page;

	if (value < 0 || value > 255)
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("trigpoint: byte value %d", value)));
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	state = GenericXLogStart(index);
	page = GenericXLogRegisterBuffer(state, buf, GENERIC_XLOG_FULL_IMAGE);
	((unsigned char *) page)[offset] = (unsigned char) value;
	GenericXLogFinish(state);
	UnlockReleaseBuffer(buf);
	relation_close(index, RowExclusiveLock);
	PG_RETURN_VOID();
}
#endif							/* TRIGPOINT_TEST_HOOKS */
