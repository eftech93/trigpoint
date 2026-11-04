/* Trigpoint access method: the shared image (E1).
 *
 * One DSM segment per (database, relfilenumber, generation) holds the generation's image in the
 * core's flat, position-independent form (csrc/cascade.cpp FlatHeader) behind a TrigpointSegHeader
 * and the TID table. A small control table — itself a named DSM segment from PostgreSQL 17's DSM
 * registry, so the module needs no shared_preload_libraries — maps the key to the segment handle.
 *
 * Protocol (tp_shared_acquire): under the table's LWLock, find the entry. Published -> attach it.
 * Being loaded by another backend -> wait on the condition variable and look again. Absent -> claim
 * a slot (loader_pid = us), release the lock, read the pages, deserialise, create the segment, write
 * the flat image into it, pin the segment for the postmaster's lifetime, publish the handle and
 * broadcast. The load runs without any LWLock held, so it may take seconds and may be cancelled; the
 * error path removes the claim and broadcasts so a waiter becomes the loader.
 *
 * Lifetime: a backend pins its mapping (dsm_pin_mapping) for as long as its cache entry lives — a
 * scan in flight on the old generation keeps its segment mapped across a compaction. The compacting
 * backend retires the old generation (tp_shared_retire): the entry is removed and the segment
 * unpinned, so the DSM control's reference count destroys it when the last mapped backend detaches
 * (its next sync sees the new generation). Entries whose relfilenumber no longer exists (DROP INDEX,
 * REINDEX) are collected the next time a backend loads an image in that database.
 *
 * Boundary rules unchanged: every PostgreSQL call here is in C; the core writes into the segment
 * through tp_index_flatten, which cannot longjmp.
 */
#include "trigpoint.h"

#include "funcapi.h"
#include "miscadmin.h"
#include "portability/instr_time.h"
#include "storage/bufmgr.h"
#include "storage/condition_variable.h"
#include "storage/ipc.h"
#include "storage/dsm.h"
#if PG_VERSION_NUM >= 170000
#include "storage/dsm_registry.h"
#else
#include "storage/ipc.h"
#include "storage/shmem.h"
#endif
#include "storage/lwlock.h"
#include "utils/builtins.h"
#include "utils/relfilenumbermap.h"
#include "storage/procarray.h"
#include "utils/wait_event.h"
#include "access/genam.h"
#include "catalog/index.h"
#include "utils/acl.h"
#include "utils/memutils.h"

#include "extension/vector/vector.h"

#define TP_SHARED_MAX 64

typedef struct TrigpointSharedEntry
{
	bool		used;
	Oid			dbid;
	Oid			spcid;
	RelFileNumber relnumber;
	uint32		generation;
	dsm_handle	handle;			/* DSM_HANDLE_INVALID while loading */
	Size		bytes;
	int			loader_pid;		/* backend loading it, 0 once published */
	TimestampTz loaded_at;
	double		t_load_ms;		/* the loader's read + deserialise + flatten */
} TrigpointSharedEntry;

typedef struct TrigpointSharedState
{
	LWLock		lock;
	ConditionVariable cv;
	TrigpointSharedEntry e[TP_SHARED_MAX];
} TrigpointSharedState;

static TrigpointSharedState *tp_state = NULL;

static void
tp_shared_init(void *ptr)
{
	TrigpointSharedState *st = (TrigpointSharedState *) ptr;

	LWLockInitialize(&st->lock, LWLockNewTrancheId());
	ConditionVariableInit(&st->cv);
	memset(st->e, 0, sizeof(st->e));
}

#if PG_VERSION_NUM >= 170000
/* 17+: the registry lives in a named DSM segment that the first backend to need it creates (no preload). */
static TrigpointSharedState *
tp_shared_state(void)
{
	bool		found;

	if (tp_state == NULL)
	{
		tp_state = (TrigpointSharedState *) GetNamedDSMSegment("trigpoint", sizeof(TrigpointSharedState), tp_shared_init, &found);
		LWLockRegisterTranche(tp_state->lock.tranche, "trigpoint");
	}
	return tp_state;
}

bool
tp_shared_available(void)
{
	return true;
}

void
tp_shared_preload_init(void)
{
}
#else
/*
 * 16: no DSM registry. Under shared_preload_libraries = 'trigpoint' the registry is carved out of the main shared
 * memory at startup (request + startup hooks) and every backend inherits the pointer; without the preload there is
 * no registry, tp_shared_available() is false and every backend keeps a private copy of each image it opens
 * (trigpoint.shared_image is then ineffective).
 */
static shmem_request_hook_type prev_shmem_request_hook = NULL;
static shmem_startup_hook_type prev_shmem_startup_hook = NULL;
static bool tp_tranche_registered = false;
static bool tp_private_logged = false;

static void
tp_shmem_request(void)
{
	if (prev_shmem_request_hook)
		prev_shmem_request_hook();
	RequestAddinShmemSpace(MAXALIGN(sizeof(TrigpointSharedState)));
}

static void
tp_shmem_startup(void)
{
	bool		found;

	if (prev_shmem_startup_hook)
		prev_shmem_startup_hook();
	LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);
	tp_state = (TrigpointSharedState *) ShmemInitStruct("trigpoint", sizeof(TrigpointSharedState), &found);
	if (!found)
		tp_shared_init(tp_state);
	LWLockRelease(AddinShmemInitLock);
}

void
tp_shared_preload_init(void)
{
	if (!process_shared_preload_libraries_in_progress)
		return;
	prev_shmem_request_hook = shmem_request_hook;
	shmem_request_hook = tp_shmem_request;
	prev_shmem_startup_hook = shmem_startup_hook;
	shmem_startup_hook = tp_shmem_startup;
}

bool
tp_shared_available(void)
{
	if (tp_state == NULL)
	{
		if (!tp_private_logged)
		{
			tp_private_logged = true;
			ereport(LOG, (errmsg("trigpoint: no shared-image registry on PostgreSQL 16 without shared_preload_libraries = 'trigpoint'; this backend keeps a private copy of each index image")));
		}
		return false;
	}
	return true;
}

static TrigpointSharedState *
tp_shared_state(void)
{
	if (tp_state != NULL && !tp_tranche_registered)
	{
		LWLockRegisterTranche(tp_state->lock.tranche, "trigpoint");
		tp_tranche_registered = true;
	}
	return tp_state;			/* NULL without the preload: callers that can run without the registry check */
}
#endif

static TrigpointSharedEntry *tp_find(TrigpointSharedState *st, Oid dbid, RelFileNumber relnumber, uint32 generation);

/* P5d: a backend that exits while it holds a claim (FATAL: pg_terminate_backend, a lost client, a recovery conflict)
 * never reaches its PG_CATCH; this exit callback clears the half-published entry so no waiter sleeps on it forever. */
static bool tp_exit_callback_registered = false;
static struct
{
	Oid			dbid;
	RelFileNumber relnumber;
	uint32		generation;
	bool		active;
}			tp_claim;

static void
tp_shared_exit_cleanup(int code, Datum arg)
{
	TrigpointSharedState *st;
	TrigpointSharedEntry *e;

	if (!tp_claim.active)
		return;
	st = tp_shared_state();
	if (st == NULL)
		return;
	LWLockAcquire(&st->lock, LW_EXCLUSIVE);
	e = tp_find(st, tp_claim.dbid, tp_claim.relnumber, tp_claim.generation);
	if (e && e->loader_pid == MyProcPid && e->handle == DSM_HANDLE_INVALID)
		memset(e, 0, sizeof(*e));
	LWLockRelease(&st->lock);
	ConditionVariableBroadcast(&st->cv);
	tp_claim.active = false;
}

static void
tp_claim_set(Oid dbid, RelFileNumber relnumber, uint32 generation)
{
	if (!tp_exit_callback_registered)
	{
		before_shmem_exit(tp_shared_exit_cleanup, (Datum) 0);
		tp_exit_callback_registered = true;
	}
	tp_claim.dbid = dbid;
	tp_claim.relnumber = relnumber;
	tp_claim.generation = generation;
	tp_claim.active = true;
}

static TrigpointSharedEntry *
tp_find(TrigpointSharedState *st, Oid dbid, RelFileNumber relnumber, uint32 generation)
{
	int			i;

	for (i = 0; i < TP_SHARED_MAX; i++)
		if (st->e[i].used && st->e[i].dbid == dbid && st->e[i].relnumber == relnumber && st->e[i].generation == generation)
			return &st->e[i];
	return NULL;
}

/* remove the entries of this database whose relation is gone (DROP INDEX, REINDEX, TRUNCATE) */
static void
tp_shared_gc(TrigpointSharedState *st)
{
	struct
	{
		Oid			spcid;
		RelFileNumber relnumber;
		uint32		generation;
		dsm_handle	handle;
	}			cand[TP_SHARED_MAX];
	int			ncand = 0,
				i;

	LWLockAcquire(&st->lock, LW_SHARED);
	for (i = 0; i < TP_SHARED_MAX; i++)
		if (st->e[i].used && st->e[i].dbid == MyDatabaseId && st->e[i].loader_pid == 0)
		{
			cand[ncand].spcid = st->e[i].spcid;
			cand[ncand].relnumber = st->e[i].relnumber;
			cand[ncand].generation = st->e[i].generation;
			cand[ncand].handle = st->e[i].handle;
			ncand++;
		}
	LWLockRelease(&st->lock);
	for (i = 0; i < ncand; i++)
	{
		if (OidIsValid(RelidByRelfilenumber(cand[i].spcid, cand[i].relnumber)))
			continue;
		LWLockAcquire(&st->lock, LW_EXCLUSIVE);
		{
			TrigpointSharedEntry *e = tp_find(st, MyDatabaseId, cand[i].relnumber, cand[i].generation);

			if (e && e->handle == cand[i].handle && e->loader_pid == 0)
			{
				dsm_unpin_segment(e->handle);
				memset(e, 0, sizeof(*e));
			}
		}
		LWLockRelease(&st->lock);
	}
}

/* P5e: one mapping per segment per backend. Two cache entries can refer to the same segment (REINDEX CONCURRENTLY
 * swaps the relfilenodes of two indexes; a stale entry pinned by a scan in flight next to its successor), and
 * dsm_attach refuses a second attach of a handle, so the mappings are counted here and detached at the last
 * release. The loader's own segment is registered the same way. */
#define TP_MAP_MAX 64
typedef struct TpMapping
{
	dsm_handle	handle;
	dsm_segment *seg;
	int			refs;
} TpMapping;
static TpMapping tp_maps[TP_MAP_MAX];
static int	tp_nmaps = 0;

static void
tp_map_add(dsm_segment *seg)
{
	if (tp_nmaps == TP_MAP_MAX)
		ereport(ERROR, (errcode(ERRCODE_INSUFFICIENT_RESOURCES), errmsg("trigpoint: too many shared images mapped in this backend (%d)", TP_MAP_MAX)));
	tp_maps[tp_nmaps].handle = dsm_segment_handle(seg);
	tp_maps[tp_nmaps].seg = seg;
	tp_maps[tp_nmaps].refs = 1;
	tp_nmaps++;
}

/* attach (and pin the mapping) once per handle; later callers share it */
static dsm_segment *
tp_map_attach(dsm_handle handle)
{
	dsm_segment *seg;
	int			i;

	for (i = 0; i < tp_nmaps; i++)
		if (tp_maps[i].handle == handle)
		{
			tp_maps[i].refs++;
			return tp_maps[i].seg;
		}
	seg = dsm_attach(handle);
	if (seg == NULL)
		return NULL;
	dsm_pin_mapping(seg);
	tp_map_add(seg);
	return seg;
}

/* the last release detaches (an unregistered segment is detached at once) */
static void
tp_map_release(dsm_segment *seg)
{
	int			i;

	for (i = 0; i < tp_nmaps; i++)
		if (tp_maps[i].seg == seg)
		{
			if (--tp_maps[i].refs > 0)
				return;
			tp_maps[i] = tp_maps[--tp_nmaps];
			break;
		}
	dsm_detach(seg);
}

/* image pages of the generation -> one malloc'd blob (tids then the core's serialised image) */
char *
tp_read_blob(Relation index, const TrigpointMetaPageData *meta)
{
	char	   *blob;
	uint64		off = 0;
	uint32		i;

	/* H3: the sizes the copies below trust must agree before any of them runs */
	if (meta->n < 0 || meta->blob_bytes != (uint64) meta->n * sizeof(uint64) + meta->core_bytes ||
		meta->core_bytes < 64 || meta->nchunks == 0 || (uint64) meta->nchunks * TP_CHUNK_BYTES < meta->blob_bytes)
		ereport(ERROR, (errcode(ERRCODE_INDEX_CORRUPTED),
						errmsg("trigpoint: meta page sizes inconsistent (n=%d image=" UINT64_FORMAT " core=" UINT64_FORMAT " pages=%u)",
							   meta->n, meta->blob_bytes, meta->core_bytes, meta->nchunks)));
	blob = (char *) malloc(meta->blob_bytes);
	if (blob == NULL)
		ereport(ERROR, (errcode(ERRCODE_OUT_OF_MEMORY), errmsg("trigpoint: out of memory loading the index image")));
	PG_TRY();
	{
		BlockNumber blk = meta->image_start;

		/* the image is a chain of pages (special `next`); since E3 they need not be contiguous */
		for (i = 0; i < meta->nchunks; i++)
		{
			Buffer		buf;
			Page		page;
			ItemId		iid;
			Size		sz;

			if (!BlockNumberIsValid(blk))
				ereport(ERROR, (errcode(ERRCODE_INDEX_CORRUPTED), errmsg("trigpoint: image chain of generation %u ends after %u of %u pages", meta->generation, i, meta->nchunks), errhint("Run trigpoint_check(<index>, true); REINDEX rebuilds the index.")));
			buf = ReadBuffer(index, blk);
			LockBuffer(buf, BUFFER_LOCK_SHARE);
			page = BufferGetPage(buf);
			if (TrigpointPageGetOpaque(page)->kind != TP_PAGE_IMAGE || TrigpointPageGetOpaque(page)->generation != meta->generation ||
				PageGetMaxOffsetNumber(page) < FirstOffsetNumber)
				ereport(ERROR, (errcode(ERRCODE_INDEX_CORRUPTED), errmsg("trigpoint: block %u is not an image page of generation %u", blk, meta->generation), errhint("Run trigpoint_check(<index>, true); REINDEX rebuilds the index.")));
			iid = PageGetItemId(page, FirstOffsetNumber);
			sz = ItemIdGetLength(iid);
			if (off + sz > meta->blob_bytes)
				ereport(ERROR, (errcode(ERRCODE_INDEX_CORRUPTED), errmsg("trigpoint: index image larger than its meta page says"), errhint("Run trigpoint_check(<index>, true); REINDEX rebuilds the index.")));
			memcpy(blob + off, PageGetItem(page, iid), sz);
			off += sz;
			blk = TrigpointPageGetOpaque(page)->next;
			UnlockReleaseBuffer(buf);
			CHECK_FOR_INTERRUPTS();
		}
		if (off != meta->blob_bytes)
			ereport(ERROR, (errcode(ERRCODE_INDEX_CORRUPTED), errmsg("trigpoint: index image truncated (" UINT64_FORMAT " of " UINT64_FORMAT " bytes)", off, meta->blob_bytes), errhint("Run trigpoint_check(<index>, true); REINDEX rebuilds the index.")));
	}
	PG_CATCH();
	{
		free(blob);
		PG_RE_THROW();
	}
	PG_END_TRY();
	return blob;
}

/* the loader: pages -> a new segment (TIDs, then the flat image at a 64-byte boundary); returns the pinned, mapped segment */
static dsm_segment *
tp_shared_load(Relation index, const TrigpointMetaPageData *meta, TrigpointCached *c, Size *bytes_out)
{
	char	   *blob = NULL;
	dsm_segment *seg = NULL;
	bool		pinned = false;
	instr_time	t0,
				t1,
				t2;

	PG_TRY();
	{
		Size		tids_bytes = (Size) meta->n * sizeof(uint64),
					hdr = MAXALIGN64(sizeof(TrigpointSegHeader)),
					tids_off,
					flat_off,
					total;
		TrigpointSegHeader *h;
		char	   *base;

		tids_off = hdr;
		flat_off = TYPEALIGN(64, tids_off + tids_bytes);
		total = flat_off + (Size) meta->core_bytes;
		if (tp_max_shared_mb > 0)
		{
			/* H8: the cap counts every image the cluster holds pinned plus this one (the sizes come from the meta page,
			 * so this runs before the pages are read); a sweep of dropped indexes first */
			uint64		in_use = tp_shared_bytes_in_use();

			if ((in_use + total) >> 20 > (uint64) tp_max_shared_mb)
			{
				tp_shared_gc(tp_shared_state());
				in_use = tp_shared_bytes_in_use();
			}
			if ((in_use + total) >> 20 > (uint64) tp_max_shared_mb)
			{
				if (tp_on_shared_cap == TP_CAP_PRIVATE)
				{
					/* P5d: the caller falls back to a private copy of the image in this backend */
					ereport(LOG, (errmsg("trigpoint: \"%s\" (%zu MB) is read as a private copy: trigpoint.max_shared_mb = %d (%llu MB in use), trigpoint.on_shared_cap = private",
										 RelationGetRelationName(index), (size_t) (total >> 20), tp_max_shared_mb, (unsigned long long) (in_use >> 20))));
					*bytes_out = 0;
					return NULL;
				}
				ereport(ERROR, (errcode(ERRCODE_CONFIGURATION_LIMIT_EXCEEDED),
								errmsg("trigpoint: loading \"%s\" (%zu MB) would exceed trigpoint.max_shared_mb = %d (%llu MB in use)",
									   RelationGetRelationName(index), (size_t) (total >> 20), tp_max_shared_mb, (unsigned long long) (in_use >> 20)),
								errhint("Raise trigpoint.max_shared_mb, drop unused trigpoint indexes, set trigpoint.on_shared_cap = private, or set trigpoint.shared_image = off for this session.")));
			}
		}
		if (tp_debug_load_sleep_ms > 0)
		{
			/* test hook: a slow load. A termination sent meanwhile lands after the load: the caller holds the meta
			 * page's share lock, an LWLock, which holds interrupts off, so the load completes and publishes a whole
			 * entry first (the terminated-loader test). */
			pg_usleep((long) tp_debug_load_sleep_ms * 1000L);
			CHECK_FOR_INTERRUPTS();
		}
		if (tp_debug_load_fatal)
			ereport(FATAL, (errmsg("trigpoint: test hook: FATAL inside the load of \"%s\"", RelationGetRelationName(index))));	/* the exit callback clears the claim */
		INSTR_TIME_SET_CURRENT(t0);
		blob = tp_read_blob(index, meta);
		INSTR_TIME_SET_CURRENT(t1);
		seg = dsm_create(total, 0);	/* ERROR (e.g. /dev/shm full) is caught below */
		base = (char *) dsm_segment_address(seg);
		h = (TrigpointSegHeader *) base;
		memset(h, 0, hdr);
		h->magic = TP_SEG_MAGIC;
		h->generation = meta->generation;
		h->dbid = MyDatabaseId;
		h->relnumber = index->rd_locator.relNumber;
		h->n = meta->n;
		h->tids_off = tids_off;
		h->flat_off = flat_off;
		h->flat_bytes = meta->core_bytes;
		h->total_bytes = total;
		memcpy(base + tids_off, blob, tids_bytes);
		memcpy(base + flat_off, blob + tids_bytes, (size_t) meta->core_bytes);
		INSTR_TIME_SET_CURRENT(t2);
		free(blob);
		blob = NULL;
		dsm_pin_segment(seg);
		pinned = true;
		dsm_pin_mapping(seg);
		tp_map_add(seg);
		c->t_read = INSTR_TIME_GET_MILLISEC(t1) - INSTR_TIME_GET_MILLISEC(t0);
		c->t_deser = 0;
		c->t_flatten = INSTR_TIME_GET_MILLISEC(t2) - INSTR_TIME_GET_MILLISEC(t1);	/* v3: the copy into the segment */
		c->loaded_here = true;
		*bytes_out = total;
	}
	PG_CATCH();
	{
		if (blob)
			free(blob);
		if (seg)
		{
			if (pinned)
				dsm_unpin_segment(dsm_segment_handle(seg));
			tp_map_release(seg);
		}
		PG_RE_THROW();
	}
	PG_END_TRY();
	return seg;
}

/*
 * Map the segment of this generation into the backend (loading it first if nobody has). Sets c->seg;
 * returns the segment header, or NULL when the cap refused the load under trigpoint.on_shared_cap = private
 * (the caller reads a private copy). The caller holds the meta page's share lock (as for an owned load), so
 * the generation cannot flip underneath; it may still have been retired before we looked (a flip
 * between our meta read and here is excluded by that lock, so the entry we publish is current).
 */
TrigpointSegHeader *
tp_shared_acquire(Relation index, const TrigpointMetaPageData *meta, TrigpointCached *c)
{
	TrigpointSharedState *st = tp_shared_state();
	Oid			dbid = MyDatabaseId;
	RelFileNumber relnumber = index->rd_locator.relNumber;
	uint32		gen = meta->generation;
	dsm_segment *seg = NULL;
	TrigpointSegHeader *h;
	instr_time	ta,
				tb;
	bool		gc_done = false;

	INSTR_TIME_SET_CURRENT(ta);
	c->loaded_here = false;
	c->t_read = c->t_deser = c->t_flatten = 0;
	for (;;)
	{
		TrigpointSharedEntry *e;
		dsm_handle	handle = DSM_HANDLE_INVALID;
		bool		loading = false,
					claim = false;
		int			i;

		CHECK_FOR_INTERRUPTS();
		LWLockAcquire(&st->lock, LW_EXCLUSIVE);
		e = tp_find(st, dbid, relnumber, gen);
		if (e != NULL)
		{
			if (e->handle != DSM_HANDLE_INVALID)
				handle = e->handle;
			else
				loading = true;
		}
		else
		{
			for (i = 0; i < TP_SHARED_MAX; i++)
				if (!st->e[i].used)
					break;
			if (i == TP_SHARED_MAX)
			{
				LWLockRelease(&st->lock);
				if (!gc_done)
				{
					tp_shared_gc(st);
					gc_done = true;
					continue;
				}
				ereport(ERROR, (errcode(ERRCODE_INSUFFICIENT_RESOURCES),
								errmsg("trigpoint: no free slot in the shared image table (%d images)", TP_SHARED_MAX),
								errhint("Drop unused trigpoint indexes or set trigpoint.shared_image = off.")));
			}
			e = &st->e[i];
			memset(e, 0, sizeof(*e));
			e->used = true;
			e->dbid = dbid;
			e->spcid = index->rd_locator.spcOid;
			e->relnumber = relnumber;
			e->generation = gen;
			e->handle = DSM_HANDLE_INVALID;
			e->loader_pid = MyProcPid;
			claim = true;
			tp_claim_set(dbid, relnumber, gen);
		}
		LWLockRelease(&st->lock);

		if (handle != DSM_HANDLE_INVALID)
		{
			seg = tp_map_attach(handle);
			if (seg == NULL)
			{
				/* destroyed between the lookup and the attach (retired): drop the stale entry, look again */
				LWLockAcquire(&st->lock, LW_EXCLUSIVE);
				e = tp_find(st, dbid, relnumber, gen);
				if (e && e->handle == handle)
					memset(e, 0, sizeof(*e));
				LWLockRelease(&st->lock);
				continue;
			}
			break;
		}
		if (loading)
		{
			/* another backend is loading this generation: wait for its broadcast, then look again; a loader that died
			 * without its exit callback (a crash of that backend alone cannot happen without a restart, but the check
			 * is cheap) is cleared here so the slot can be claimed again */
			int			loader = 0;

			ConditionVariablePrepareToSleep(&st->cv);
			LWLockAcquire(&st->lock, LW_SHARED);
			e = tp_find(st, dbid, relnumber, gen);
			loading = (e != NULL && e->handle == DSM_HANDLE_INVALID);
			if (loading)
				loader = e->loader_pid;
			LWLockRelease(&st->lock);
			if (loading && loader != 0 && loader != MyProcPid && BackendPidGetProc(loader) == NULL)
			{
				LWLockAcquire(&st->lock, LW_EXCLUSIVE);
				e = tp_find(st, dbid, relnumber, gen);
				if (e && e->handle == DSM_HANDLE_INVALID && e->loader_pid == loader)
				{
					memset(e, 0, sizeof(*e));
					ereport(LOG, (errmsg("trigpoint: cleared the half-published image of \"%s\" generation %u (loader %d is gone)",
										 RelationGetRelationName(index), gen, loader)));
				}
				LWLockRelease(&st->lock);
				loading = false;
			}
			if (loading)
				ConditionVariableTimedSleep(&st->cv, 1000, PG_WAIT_EXTENSION);
			ConditionVariableCancelSleep();
			continue;
		}
		if (claim)
		{
			Size		bytes = 0;

			if (!gc_done)
			{
				tp_shared_gc(st);
				gc_done = true;
			}
			PG_TRY();
			{
				seg = tp_shared_load(index, meta, c, &bytes);
			}
			PG_CATCH();
			{
				LWLockAcquire(&st->lock, LW_EXCLUSIVE);
				e = tp_find(st, dbid, relnumber, gen);
				if (e && e->loader_pid == MyProcPid)
					memset(e, 0, sizeof(*e));
				LWLockRelease(&st->lock);
				tp_claim.active = false;
				ConditionVariableBroadcast(&st->cv);
				PG_RE_THROW();
			}
			PG_END_TRY();
			LWLockAcquire(&st->lock, LW_EXCLUSIVE);
			e = tp_find(st, dbid, relnumber, gen);
			if (seg == NULL)
			{
				/* the cap, with trigpoint.on_shared_cap = private: give the slot back, the caller reads a private copy */
				if (e && e->loader_pid == MyProcPid)
					memset(e, 0, sizeof(*e));
				LWLockRelease(&st->lock);
				tp_claim.active = false;
				ConditionVariableBroadcast(&st->cv);
				return NULL;
			}
			if (e && e->loader_pid == MyProcPid)
			{
				e->handle = dsm_segment_handle(seg);
				e->bytes = bytes;
				e->loader_pid = 0;
				e->loaded_at = GetCurrentTimestamp();
				e->t_load_ms = c->t_read + c->t_deser + c->t_flatten;
				LWLockRelease(&st->lock);
				tp_claim.active = false;
				ConditionVariableBroadcast(&st->cv);
				break;
			}
			/* our claim vanished (the generation was retired while we loaded): discard and start over */
			LWLockRelease(&st->lock);
			tp_claim.active = false;
			dsm_unpin_segment(dsm_segment_handle(seg));
			tp_map_release(seg);
			seg = NULL;
			continue;
		}
	}
	h = (TrigpointSegHeader *) dsm_segment_address(seg);
	if (h->magic != TP_SEG_MAGIC || h->generation != gen || h->relnumber != relnumber || h->n != meta->n)
	{
		tp_map_release(seg);
		ereport(ERROR, (errcode(ERRCODE_INDEX_CORRUPTED), errmsg("trigpoint: shared image segment does not match the index (generation %u)", gen), errhint("Run trigpoint_check(<index>, true); REINDEX rebuilds the index.")));
	}
	c->seg = seg;
	INSTR_TIME_SET_CURRENT(tb);
	c->t_attach = INSTR_TIME_GET_MILLISEC(tb) - INSTR_TIME_GET_MILLISEC(ta) - (c->t_read + c->t_deser + c->t_flatten);
	return h;
}

void
tp_shared_release(TrigpointCached *c)
{
	if (c->seg)
	{
		tp_map_release((dsm_segment *) c->seg);	/* the last detach of an unpinned (retired) segment destroys it */
		c->seg = NULL;
	}
	c->tids_base = NULL;
	c->n_base = 0;
}

/* a compaction replaced generation `generation`: forget its segment (destroyed when the last backend detaches) */
void
tp_shared_retire(Relation index, uint32 generation)
{
	TrigpointSharedState *st = tp_shared_state();
	TrigpointSharedEntry *e;

	if (st == NULL)
		return;
	LWLockAcquire(&st->lock, LW_EXCLUSIVE);
	e = tp_find(st, MyDatabaseId, index->rd_locator.relNumber, generation);
	if (e)
	{
		if (e->handle != DSM_HANDLE_INVALID)
			dsm_unpin_segment(e->handle);
		memset(e, 0, sizeof(*e));	/* a loader of this generation finds its claim gone and discards its segment */
	}
	LWLockRelease(&st->lock);
	ConditionVariableBroadcast(&st->cv);
}

/* ------------------------------------------------------------------------------------------------ */
/* SQL: trigpoint_shared() -> one row per shared image (this cluster), trigpoint_backend_memory() */

PG_FUNCTION_INFO_V1(trigpoint_shared);
PG_FUNCTION_INFO_V1(trigpoint_backend_memory);
PG_FUNCTION_INFO_V1(trigpoint_image);
PG_FUNCTION_INFO_V1(trigpoint_route);

/* trigpoint_image(regclass) -> bytea: the current generation's blob (TID table, then the flat image) as the
 * pages hold it — for the parity gate and routing experiments in Python (75k / 100k images; bytea <= 1 GB). */
Datum
trigpoint_image(PG_FUNCTION_ARGS)
{
	Oid			relid = PG_GETARG_OID(0);
	Relation	index = tp_open_index_checked(relid, AccessShareLock, false);
	TrigpointMetaPageData meta;
	char	   *blob;
	bytea	   *out;

	tp_read_meta(index, &meta);
	if (meta.blob_bytes > (uint64) MaxAllocSize - VARHDRSZ)
		ereport(ERROR, (errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED), errmsg("trigpoint: image too large for a bytea (" UINT64_FORMAT " bytes)", meta.blob_bytes)));
	blob = tp_read_blob(index, &meta);
	out = (bytea *) palloc((Size) meta.blob_bytes + VARHDRSZ);
	SET_VARSIZE(out, meta.blob_bytes + VARHDRSZ);
	memcpy(VARDATA(out), blob, (size_t) meta.blob_bytes);
	free(blob);
	index_close(index, AccessShareLock);
	PG_RETURN_BYTEA_P(out);
}

/* trigpoint_route(regclass, vector, M, ef) -> setof (tid, score): the router's top-M hits for a query (parity gate) */
Datum
trigpoint_route(PG_FUNCTION_ARGS)
{
	ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
	Oid			relid = PG_GETARG_OID(0);
	Vector	   *q = (Vector *) PG_DETOAST_DATUM(PG_GETARG_DATUM(1));
	int			M = PG_GETARG_INT32(2),
				ef = PG_GETARG_INT32(3);
	Relation	index = tp_open_index_checked(relid, AccessShareLock, false);
	TrigpointCached *c;
	int		   *ids;
	float	   *scores;
	int			k,
				i;

	InitMaterializedSRF(fcinfo, 0);
	c = tp_cache_get(index);
	PG_TRY();
	{
		if (q->dim != c->D)
			ereport(ERROR, (errcode(ERRCODE_DATA_EXCEPTION), errmsg("trigpoint: query has %d dimensions, index has %d", q->dim, c->D)));
		ids = (int *) palloc(sizeof(int) * (Size) Max(M, 1));
		scores = (float *) palloc(sizeof(float) * (Size) Max(M, 1));
		k = tp_route_hits(c->ix, q->x, M, ef, ids, scores);
		if (k < 0)
			ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("trigpoint: %s", tp_last_error())));
		for (i = 0; i < k; i++)
		{
			Datum		values[2];
			bool		nulls[2] = {false, false};
			ItemPointerData tid;
			ItemPointer tidp;

			if (!tp_tid_of(c, ids[i], &tid))
				continue;
			tidp = (ItemPointer) palloc(sizeof(ItemPointerData));
			*tidp = tid;
			values[0] = PointerGetDatum(tidp);
			values[1] = Float4GetDatum(scores[i]);
			tuplestore_putvalues(rsinfo->setResult, rsinfo->setDesc, values, nulls);
		}
	}
	PG_CATCH();
	{
		tp_cache_unpin(c);
		PG_RE_THROW();
	}
	PG_END_TRY();
	tp_cache_unpin(c);
	index_close(index, AccessShareLock);
	return (Datum) 0;
}

/* bytes of every published image (H8) */
uint64
tp_shared_bytes_in_use(void)
{
	TrigpointSharedState *st = tp_shared_state();
	uint64		sum = 0;
	int			i;

	if (st == NULL)
		return 0;
	LWLockAcquire(&st->lock, LW_SHARED);
	for (i = 0; i < TP_SHARED_MAX; i++)
		if (st->e[i].used && st->e[i].handle != DSM_HANDLE_INVALID)
			sum += (uint64) st->e[i].bytes;
	LWLockRelease(&st->lock);
	return sum;
}

PG_FUNCTION_INFO_V1(trigpoint_shared_available);

/* SQL: whether images can be shared on this server (true on 17+; on 16 only under shared_preload_libraries) */
Datum
trigpoint_shared_available(PG_FUNCTION_ARGS)
{
	PG_RETURN_BOOL(tp_shared_available());
}

PG_FUNCTION_INFO_V1(trigpoint_shared_total);

Datum
trigpoint_shared_total(PG_FUNCTION_ARGS)
{
	TupleDesc	tupdesc;
	Datum		values[4];
	bool		nulls[4] = {false, false, false, false};
	TrigpointSharedState *st = tp_shared_state();
	int			images = 0,
				loading = 0,
				i;

	if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
		elog(ERROR, "return type must be a row type");
	if (st != NULL)
		LWLockAcquire(&st->lock, LW_SHARED);
	for (i = 0; st != NULL && i < TP_SHARED_MAX; i++)
		if (st->e[i].used)
		{
			images++;
			if (st->e[i].handle == DSM_HANDLE_INVALID)
				loading++;
		}
	if (st != NULL)
		LWLockRelease(&st->lock);
	values[0] = Int32GetDatum(images);
	values[1] = Int32GetDatum(loading);
	values[2] = Int64GetDatum((int64) tp_shared_bytes_in_use());
	values[3] = Int32GetDatum(tp_max_shared_mb);
	PG_RETURN_DATUM(HeapTupleGetDatum(heap_form_tuple(tupdesc, values, nulls)));
}

Datum
trigpoint_shared(PG_FUNCTION_ARGS)
{
	ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
	TrigpointSharedState *st = tp_shared_state();
	TrigpointSharedEntry snap[TP_SHARED_MAX];
	int			i;

	InitMaterializedSRF(fcinfo, 0);
	if (st == NULL)
		return (Datum) 0;		/* no registry (16 without the preload): no shared images */
	LWLockAcquire(&st->lock, LW_SHARED);
	memcpy(snap, st->e, sizeof(snap));
	LWLockRelease(&st->lock);
	for (i = 0; i < TP_SHARED_MAX; i++)
	{
		Datum		values[8];
		bool		nulls[8] = {false, false, false, false, false, false, false, false};
		Oid			relid;

		if (!snap[i].used)
			continue;
		relid = (snap[i].dbid == MyDatabaseId) ? RelidByRelfilenumber(snap[i].spcid, snap[i].relnumber) : InvalidOid;
		if (!superuser())
		{
			/* a row per index the caller may read; other databases' images and unresolvable entries are superuser-only */
			Oid			heapid = OidIsValid(relid) ? IndexGetRelation(relid, true) : InvalidOid;

			if (!OidIsValid(heapid) || pg_class_aclcheck(heapid, GetUserId(), ACL_SELECT) != ACLCHECK_OK)
				continue;
		}
		values[0] = ObjectIdGetDatum(snap[i].dbid);
		values[1] = ObjectIdGetDatum(snap[i].relnumber);
		if (OidIsValid(relid))
			values[2] = ObjectIdGetDatum(relid);
		else
			nulls[2] = true;
		values[3] = Int64GetDatum((int64) snap[i].generation);
		values[4] = Int64GetDatum((int64) snap[i].bytes);
		values[5] = BoolGetDatum(snap[i].loader_pid != 0);
		if (snap[i].loader_pid == 0)
			values[6] = TimestampTzGetDatum(snap[i].loaded_at);
		else
			nulls[6] = true;
		values[7] = Float8GetDatum(snap[i].t_load_ms);
		tuplestore_putvalues(rsinfo->setResult, rsinfo->setDesc, values, nulls);
	}
	return (Datum) 0;
}

/* this backend's memory as the kernel sees it: /proc/self/status Vm/Rss lines, one text */
Datum
trigpoint_backend_memory(PG_FUNCTION_ARGS)
{
	StringInfoData s;
	FILE	   *f = fopen("/proc/self/status", "r");
	char		line[256];

	initStringInfo(&s);
	if (f == NULL)
		PG_RETURN_TEXT_P(cstring_to_text("unavailable"));
	while (fgets(line, sizeof(line), f))
	{
		if (strncmp(line, "VmRSS:", 6) == 0 || strncmp(line, "RssAnon:", 8) == 0 || strncmp(line, "RssFile:", 8) == 0 ||
			strncmp(line, "RssShmem:", 9) == 0 || strncmp(line, "VmHWM:", 6) == 0)
		{
			char	   *p = line,
					   *q;

			while (*p && *p != ':')
				p++;
			if (*p == ':')
				*p++ = 0;
			while (*p == ' ' || *p == '\t')
				p++;
			q = p + strlen(p);
			while (q > p && (q[-1] == '\n' || q[-1] == ' '))
				*--q = 0;
			appendStringInfo(&s, "%s%s=%s", s.len ? " " : "", line, p);
		}
	}
	fclose(f);
	PG_RETURN_TEXT_P(cstring_to_text(s.data));
}
