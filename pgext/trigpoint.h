/* Trigpoint index access method: shared declarations (spike S2). */
#ifndef TRIGPOINT_H
#define TRIGPOINT_H

#include "postgres.h"

#include "access/amapi.h"
#include "access/genam.h"
#include "access/reloptions.h"
#include "access/relscan.h"
#include "nodes/execnodes.h"
#include "nodes/pathnodes.h"
#include "storage/bufmgr.h"
#include "storage/bufpage.h"
#include "storage/itemptr.h"
#include "utils/rel.h"
#include "storage/dsm.h"
#include "storage/shm_toc.h"

#include "trigpoint_core.h"

#include <math.h>

#define TP_MAGIC			0x58495054	/* "TPIX" */
/* On-disk format policy. TP_META_VERSION 3 is read by every 0.x and 1.x release and is the
 * same on every PostgreSQL major: the page structs below use fixed-width fields only, MAXALIGN is 8 on every 64-bit
 * platform and pg_upgrade copies index files verbatim within one architecture and BLCKSZ (scripts/pg_upgrade_test.sh
 * checks 17 -> 18). A reader refuses any other version with a REINDEX hint; a bump ships an in-place migration or a
 * documented REINDEX. Version 2 was the spike's serialised core. */
#define TP_META_VERSION		3	/* E2: pages hold the TID table + the flat image */
#define TP_METAPAGE_BLKNO	0
/* one item per image page: BLCKSZ - page header (24) - special (16) - one line pointer (4), MAXALIGNed */
#define TP_CHUNK_BYTES		8144

/* every non-meta page carries this in its special space */
#define TP_PAGE_IMAGE		1		/* a chunk of a serialised index image (one item) */
#define TP_PAGE_LOG			2		/* insert / tombstone records appended since the image was written */
typedef struct TrigpointPageOpaqueData
{
	uint16		kind;
	uint16		unused;
	uint32		generation;		/* image generation this page belongs to */
	BlockNumber next;			/* log pages: the next log page, InvalidBlockNumber at the tail */
} TrigpointPageOpaqueData;
typedef TrigpointPageOpaqueData *TrigpointPageOpaque;
#define TrigpointPageGetOpaque(page) ((TrigpointPageOpaque) PageGetSpecialPointer(page))

/*
 * Log records. An INSERT record carries everything a backend needs to add the row to its copy of
 * the index without the vector: the cell, the int8 codes (d3 bytes) and the per-level norms; its
 * id is implicit (n0 + ordinal of the record in the log). A DELETE record tombstones an id.
 */
#define TP_REC_INSERT		1
#define TP_REC_DELETE		2
typedef struct TrigpointLogRecord
{
	uint16		kind;
	uint16		cell;
	int32		id;				/* DELETE: the id; INSERT: the id assigned (n0 + ordinal), for checking */
	uint64		tid;			/* INSERT: heap TID as block << 16 | offset */
	float		norms[8];		/* INSERT: err[L] then tail[L] */
	int8		codes[FLEXIBLE_ARRAY_MEMBER];	/* INSERT: d3 bytes */
} TrigpointLogRecord;
#define TP_INSERT_RECORD_SIZE(d3) MAXALIGN(offsetof(TrigpointLogRecord, codes) + (d3))
#define TP_DELETE_RECORD_SIZE MAXALIGN(offsetof(TrigpointLogRecord, codes))

/*
 * Block 0. Blocks 1..nchunks hold the index image as one item each: n TIDs (uint64, block << 16 |
 * offset) followed by the core's serialised image (codes, norms, router graph, calibration). The
 * image is read once per backend into a malloc'd cache (tp_cache_get) and searched from memory;
 * the cell / router page kinds of the plan are sections of this image, not yet pages of their own.
 */
typedef struct TrigpointMetaPageData
{
	uint32		magic;
	uint32		version;
	int32		n;				/* rows indexed */
	int32		D;				/* vector dimension */
	int32		C;				/* cells */
	int32		L;				/* levels */
	int32		lv[4];			/* cumulative level widths, lv[L-1] = sketch width */
	int32		M;				/* router graph degree parameter */
	int32		efC;			/* router graph ef_construction */
	uint32		nchunks;		/* image pages of the current generation */
	uint64		blob_bytes;		/* tids + core image */
	uint64		core_bytes;		/* core image alone */
	/* S4: generations and the log. The image of generation g starts at image_start (nchunks pages,
	 * not necessarily contiguous with the meta page after a compaction); inserts and tombstones
	 * since then are appended to a chain of log pages log_first .. log_last. A compaction
	 * (amvacuumcleanup) writes the next generation's image into new pages and flips these fields
	 * under the meta page's exclusive lock; readers hold its share lock while they load. */
	uint32		generation;
	BlockNumber image_start;
	BlockNumber log_first;
	BlockNumber log_last;
	int64		log_inserts;	/* records since the image was written (for the compaction trigger) */
	int64		log_deletes;
	int32		n_image;		/* rows in the image (n0 of the core index) */
	int32		compactions;
	float		ch_default[4];	/* calibrated tail cosines at delta = 0.01 */
	float		var_captured;	/* share of the sample variance in the head */
	double		t_scan;			/* build timings, seconds */
	double		t_fit;
	double		t_head;
	double		t_kmeans;
	double		t_index;
	double		t_calib;
	double		t_write;
	double		t_codes;		/* E2: the codes and the graph separately (t_index = both) */
	double		t_graph;
	int32		participants;	/* E2: workers + leader of the build (0: the core's thread path) */
	int32		metric;			/* H11: TP_METRIC_L2 or TP_METRIC_COSINE (rows and queries normalised; the order is cosine's) */
} TrigpointMetaPageData;

typedef TrigpointMetaPageData *TrigpointMetaPage;

/* H11: metrics. Cosine distance orders like L2 on unit vectors, so the index stores every row normalised,
 * normalises the query, and ranks fetched rows by the squared L2 distance of their normalised forms. */
#define TP_METRIC_L2		0
#define TP_METRIC_COSINE	1
extern int	tp_index_metric(Relation index);		/* from the index's operator family */
static inline void
tp_normalize(float *x, int D)
{
	double		s = 0;
	int			i;

	for (i = 0; i < D; i++)
		s += (double) x[i] * x[i];
	if (s > 0)
	{
		float		inv = (float) (1.0 / sqrt(s));

		for (i = 0; i < D; i++)
			x[i] *= inv;
	}
}

/*
 * Read a block reached through a pointer stored in the index (a chain's `next`). A pointer at or past the end of the
 * relation is index corruption (XX002), not the storage manager's could-not-open-file: *nblocks caches the relation's
 * length (0 = not read yet) and is re-read once before giving up, since appenders extend the log meanwhile.
 */
static inline Buffer
tp_read_linked(Relation index, BlockNumber blk, BlockNumber *nblocks)
{
	if (blk >= *nblocks)
	{
		*nblocks = RelationGetNumberOfBlocks(index);
		if (blk >= *nblocks)
			ereport(ERROR, (errcode(ERRCODE_INDEX_CORRUPTED),
							errmsg("trigpoint: index \"%s\" points at block %u, past its %u blocks", RelationGetRelationName(index), blk, *nblocks),
							errhint("Run trigpoint_check(<index>, true); REINDEX rebuilds the index.")));
	}
	return ReadBuffer(index, blk);
}

/*
 * A page's header is ours and sane before anything reads through it: the special space is our size and where it
 * should be, and pd_lower <= pd_upper <= pd_special (PageGetMaxOffsetNumber and PageGetSpecialPointer read past the
 * page otherwise). Then an item fits when it is a normal item inside [pd_upper, pd_special).
 */
static inline bool
tp_page_header_ok(Page page)
{
	PageHeader	ph = (PageHeader) page;

	return !PageIsNew(page) && ph->pd_special == BLCKSZ - MAXALIGN(sizeof(TrigpointPageOpaqueData)) &&
		ph->pd_lower >= SizeOfPageHeaderData && ph->pd_lower <= ph->pd_upper && ph->pd_upper <= ph->pd_special;
}

static inline bool
tp_item_fits(Page page, ItemId iid)
{
	PageHeader	ph = (PageHeader) page;

	return ItemIdIsNormal(iid) && ItemIdGetOffset(iid) >= ph->pd_upper &&
		ItemIdGetOffset(iid) + ItemIdGetLength(iid) <= ph->pd_special;
}

/* tp_scan.c: the checked log record at (page, off) — ERROR (index corrupted) when it is not one (H3) */
extern char *tp_meta_problem(const TrigpointMetaPageData *meta, BlockNumber nblocks);
struct TrigpointLogRecord;
extern struct TrigpointLogRecord *tp_log_record(Page page, OffsetNumber off, BlockNumber blk, int D, int L, const int32 *lv, int C);
extern struct TrigpointLogRecord *tp_log_record_problem(Page page, OffsetNumber off, BlockNumber blk, int L, const int32 *lv, int C, char **why);

#define TrigpointPageGetMeta(page) ((TrigpointMetaPage) PageGetContents(page))

typedef struct TrigpointOptions
{
	int32		vl_len_;
	int			cells;			/* 0 = automatic */
	int			m;
	int			ef_construction;
	int			sample;
	int			kmeans_iters;
	int			calibration_queries;
	int			threads;		/* build threads inside the core (1 = serial; honoured only with trigpoint.allow_build_threads) */
	int			seed;			/* PCA sample, k-means init and router levels (E2: deterministic builds) */
	int			levels_offset;	/* string reloption, "" = automatic by dimension */
} TrigpointOptions;

/* GUCs (trigpoint.c) */
extern int	tp_probes;
extern int	tp_hits;
extern int	tp_ef;
extern int	tp_k;
enum { TP_ITER_OFF = 0, TP_ITER_RELAXED = 1, TP_ITER_STRICT = 2 };	/* trigpoint.iterative_scan */
extern int	tp_iterative;
extern int	tp_max_scan_tuples;
extern int	tp_max_probes;
extern double tp_delta;
extern bool tp_exact;
extern bool tp_merge;
extern int	tp_seed_cells;
extern double tp_compact_frac;
extern bool tp_compact_reclaim;
extern int tp_recalibrate_queries;
extern int tp_compact_records;
extern int tp_max_shared_mb;
enum { TP_CAP_ERROR = 0, TP_CAP_PRIVATE = 1 };	/* trigpoint.on_shared_cap */
extern int	tp_on_shared_cap;
extern bool tp_debug;
#ifdef TRIGPOINT_TEST_HOOKS
extern bool tp_debug_skip_heap;	/* test builds only: these change results */
extern bool tp_debug_no_delta_graph;
extern int	tp_debug_load_sleep_ms;
extern bool tp_debug_load_fatal;
#else
#define tp_debug_skip_heap false
#define tp_debug_no_delta_graph false
#define tp_debug_load_sleep_ms 0
#define tp_debug_load_fatal false
#endif
extern int tp_delta_flat_max;
extern int tp_recalibrate_probes;
extern bool tp_shared;
extern bool tp_shared_available(void);	/* the cluster-wide image registry exists (always on 17+; on 16 only under shared_preload_libraries) */
extern void tp_shared_preload_init(void);
static inline bool
tp_shared_effective(void)
{
	return tp_shared && tp_shared_available();
}
#ifdef TRIGPOINT_TEST_HOOKS
extern bool tp_allow_build_threads;
#else
#define tp_allow_build_threads false
#endif
extern relopt_kind tp_relopt_kind;

/*
 * Per-backend cache of opened indexes (tp_scan.c). Two forms of one entry:
 *   owned  (trigpoint.shared_image = off, and always for VACUUM's compaction): the generation's image
 *          is deserialised into this backend's malloc'd core index; tids holds every id;
 *   shared (E1, default): the generation's image lives flat in a DSM segment (tp_shared.c), loaded
 *          once per postmaster lifetime by the first backend that needs it; this backend maps it
 *          (seg), the core reads it in place (tp_index_open_flat) and keeps only the log since the
 *          image — inserts, tombstones — in private memory. tids_base points at the segment's TID
 *          table (ids < n_base), tids holds the ids appended since (id - n_base).
 */
typedef struct TrigpointCached
{
	Oid			relid;
	void	   *ix;				/* core index (malloc'd by the shim) */
	ItemPointerData *tids;		/* owned: id -> heap TID for every id; shared: for ids >= n_base (malloc'd) */
	const uint64 *tids_base;	/* shared: the segment's TID table (block << 16 | offset, 0 = tombstoned at image time) */
	int			n_base;			/* shared: ids covered by tids_base (0 when owned) */
	void	   *seg;			/* shared: dsm_segment * this backend has mapped */
	char	   *priv_image;		/* owned: this backend's malloc'd copy of the pages' blob (TIDs + flat image) */
	bool		owned;			/* this entry was opened owned (VACUUM's transient copy or shared_image = off) */
	void	   *dgraph;			/* FlatGraph * over the positions appended since the image (malloc'd, this backend's) */
	void	   *dscratch;		/* TpGraphScratch * for inserting into it */
	int			n;
	int			D;
	int			C;
	int			L;
	int			pins;			/* scans using it */
	bool		stale;			/* invalidated while pinned: free at unpin */
	TrigpointMetaPageData meta;
	/* how far this copy has applied the log: records up to (log_blk, log_off) inclusive */
	uint32		generation;
	BlockNumber log_blk;
	OffsetNumber log_off;
	int			n_cap;			/* tids capacity */
	/* timings of the last load (ms), for trigpoint.debug and the E1 measurements */
	double		t_read;			/* image pages -> malloc'd blob */
	double		t_deser;		/* blob -> owned core index */
	double		t_flatten;		/* owned index -> flat image in the segment (the loading backend only) */
	double		t_attach;		/* dsm_attach + tp_index_open_flat */
	double		t_replay;		/* log records applied at load */
	int			replayed;
	int			skipped;		/* INSERT records whose DELETE was in the same replay: applied, tombstoned, not graphed */
	bool		loaded_here;	/* this backend created the segment */
	/* set only while apply_log_from replays with skip_cancelled: dead[id - skip_base] for the INSERT ids ahead */
	const uint8 *skip_dead;
	int			skip_base;
	int			skip_n;
} TrigpointCached;

/* shared image segments (tp_shared.c) */
typedef struct TrigpointSegHeader
{
	uint32		magic;
	uint32		generation;
	Oid			dbid;
	RelFileNumber relnumber;
	int32		n;				/* ids in the TID table */
	uint32		unused;
	uint64		tids_off;		/* uint64 TIDs, n of them */
	uint64		flat_off;		/* the core's flat image */
	uint64		flat_bytes;
	uint64		total_bytes;
} TrigpointSegHeader;
#define TP_SEG_MAGIC 0x47455354	/* "TSEG" */

extern TrigpointSegHeader *tp_shared_acquire(Relation index, const TrigpointMetaPageData *meta, TrigpointCached *c);
extern void tp_shared_release(TrigpointCached *c);
extern void tp_shared_retire(Relation index, uint32 generation);
extern void tp_shared_collect(void);	/* unpin the images of this database's dropped indexes */
extern char *tp_read_blob(Relation index, const TrigpointMetaPageData *meta);
extern void tp_delta_graph_free(TrigpointCached *c);
extern void tp_delta_graph_add(TrigpointCached *c, int k, const int8 *codes);

extern TrigpointCached *tp_cache_get(Relation index);
extern TrigpointCached *tp_cache_get_owned(Relation index);	/* transient owned copy (compaction) */
extern void tp_cache_unpin(TrigpointCached *c);
extern uint64 tp_shared_bytes_in_use(void);	/* H8 */
extern void tp_write_trivial(Relation index, ForkNumber fork, TrigpointMetaPageData *meta);	/* H14 */
extern bool tp_tid_of(TrigpointCached *c, int id, ItemPointer out);	/* false when tombstoned / unknown */
extern void tp_cache_apply_record(TrigpointCached *c, TrigpointLogRecord *rec);
extern void tp_read_meta(Relation index, TrigpointMetaPageData *meta);
extern void tp_cache_sync(Relation index, TrigpointCached *c);
extern void tp_log_append(Relation index, TrigpointCached *c, TrigpointLogRecord *rec, Size size);
extern void tp_write_image_pages(Relation index, ForkNumber fork, TrigpointMetaPageData *meta, const char *blob, BlockNumber *start_out);
extern Buffer tp_new_page(Relation index, ForkNumber fork);	/* E3: a free page from the FSM, else a new one; exclusively locked */
extern void tp_write_index2(Relation index, ForkNumber fork, TrigpointMetaPageData *meta, const char *p1, uint64 l1, const char *p2, uint64 l2, BlockNumber *start_out);
extern int	tp_auto_cells(int64 n);
extern int	tp_cmp_float(const void *a, const void *b);
/* the parallel build (tp_pbuild.c) */
extern int64 tp_parallel_build(Relation heap, Relation index, IndexInfo *indexInfo, TrigpointMetaPageData *meta,
							   const TrigpointOptions *opts, int request, int *nparticipants_out, double *phase_s);
extern PGDLLEXPORT void tp_parallel_build_main(dsm_segment *seg, shm_toc *toc);	/* looked up by name in the workers: must stay exported */
extern void tp_init_page(Page page, Size size, uint16 kind, uint32 generation);

/* access method functions */
extern IndexBuildResult *tp_build(Relation heap, Relation index, IndexInfo *indexInfo);
/* SQL-callable helpers open an index through this: lock, then relkind/AM check, then the caller's SELECT privilege on
 * the indexed heap (or ownership when need_owner) — index ACLs are never checked by PostgreSQL itself. */
extern Relation tp_open_index_checked(Oid relid, LOCKMODE lockmode, bool need_owner);
extern void tp_buildempty(Relation index);
extern bool tp_insert(Relation index, Datum *values, bool *isnull, ItemPointer heap_tid, Relation heap,
					  IndexUniqueCheck checkUnique, bool indexUnchanged, IndexInfo *indexInfo);
extern IndexBulkDeleteResult *tp_bulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats,
											IndexBulkDeleteCallback callback, void *callback_state);
extern IndexBulkDeleteResult *tp_vacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats);
extern bytea *tp_options(Datum reloptions, bool validate);
extern bool tp_validate(Oid opclassoid);
extern void tp_costestimate(PlannerInfo *root, IndexPath *path, double loop_count, Cost *indexStartupCost,
							Cost *indexTotalCost, Selectivity *indexSelectivity, double *indexCorrelation,
							double *indexPages);
extern IndexScanDesc tp_beginscan(Relation index, int nkeys, int norderbys);
extern void tp_rescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys);
extern bool tp_gettuple(IndexScanDesc scan, ScanDirection dir);
extern void tp_endscan(IndexScanDesc scan);

#endif
