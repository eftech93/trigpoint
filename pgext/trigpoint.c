/* Trigpoint — PostgreSQL index access method for the sketch-routed, calibrated-cascade vector index.
 *
 * This file: module
 * init (GUCs, reloptions, relcache callback), the AM handler and the SQL-visible helper functions.
 * tp_build.c: ambuild and the write side; tp_scan.c: the per-backend cache and the read path.
 * The C++ core (csrc/cascade.cpp) is linked in as a static library behind trigpoint_core.h; no
 * PostgreSQL call runs inside a C++ frame and no C++ exception reaches the backend.
 */
#include "trigpoint.h"

#include <limits.h>

#include "access/relation.h"
#include "catalog/index.h"
#include "catalog/pg_am.h"
#include "utils/syscache.h"
#include "catalog/pg_opfamily.h"
#include "utils/rel.h"
#include "nodes/value.h"
#include "nodes/makefuncs.h"
#include "commands/defrem.h"
#include "commands/vacuum.h"
#include "storage/bufmgr.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/lsyscache.h"
#include "utils/acl.h"
#include "catalog/pg_class.h"

PG_MODULE_MAGIC;

int			tp_probes = 32;		/* P: cells scanned */
int			tp_hits = 200;		/* M: router hits */
int			tp_ef = 0;			/* router beam width, 0 = hits */
int			tp_k = 10;			/* neighbours computed per scan */
int			tp_iterative = TP_ITER_RELAXED;	/* how a scan goes on when the executor asks for more than its first k */
int			tp_max_scan_tuples = 20000;	/* iterative scan: heap rows fetched before the scan stops */
int			tp_max_probes = 0;			/* iterative scan: cells opened before the scan stops (0: every cell) */
static const struct config_enum_entry tp_iterative_options[] = {
	{"off", TP_ITER_OFF, false},
	{"relaxed_order", TP_ITER_RELAXED, false},
	{"strict_order", TP_ITER_STRICT, false},
	{NULL, 0, false}
};
double		tp_delta = 0.01;	/* miss probability of the calibrated bound */
bool		tp_exact = false;	/* certificate: exact bound, every cell */
bool		tp_merge = true;	/* after the seed cells, fetch the remaining cells' survivors in one global bound order */
int			tp_seed_cells = 1;	/* cells scanned cell-by-cell (tightening tau) before the merged rest */
double		tp_compact_frac = 0.2;
bool		tp_compact_reclaim = true;	/* E3b: after a compaction, move the image into the freed pages and truncate the relation */	/* VACUUM compacts when log inserts + tombstones exceed this share of the image */
bool		tp_debug = false;	/* NOTICE with per-query statistics */
int			tp_recalibrate_queries = 500;	/* H16 */
int			tp_delta_flat_max = 50000;	/* H17 */
int			tp_recalibrate_probes = 64;	/* H18 */
int			tp_compact_records = 20000;	/* H9: VACUUM compacts when the log holds more records than this, whatever the fraction */
int			tp_max_shared_mb = 0;			/* H8: cap on the pinned shared images of the cluster, 0 = none */
int			tp_on_shared_cap = TP_CAP_ERROR;	/* P5d: at the cap, refuse the load (error) or keep a private copy (private) */
static const struct config_enum_entry tp_on_shared_cap_options[] = {
	{"error", TP_CAP_ERROR, false},
	{"private", TP_CAP_PRIVATE, false},
	{NULL, 0, false}
};
#ifdef TRIGPOINT_TEST_HOOKS
int			tp_debug_load_sleep_ms = 0;	/* test hook: sleep inside a shared-image load while the slot is claimed (the killed-loader test) */
bool		tp_debug_load_fatal = false;	/* test hook: raise FATAL inside a shared-image load while the slot is claimed (the exit-callback test) */
#endif
#ifdef TRIGPOINT_TEST_HOOKS
bool		tp_debug_skip_heap = false;	/* diagnostic: no heap fetches, results ranked by lower bound */
bool		tp_debug_no_delta_graph = false;	/* diagnostic: route without the delta graph */
bool		tp_debug_centroid_router = false;	/* ablation: rank cells by centroid distance alone, no router graph */
double		tp_debug_ch = 0;	/* ablation: >0 replaces the calibrated tail cosines by this value at every level */
#endif
bool		tp_shared = true;	/* E1: read the generation's image from one DSM segment shared by every backend */
#ifdef TRIGPOINT_TEST_HOOKS
bool		tp_allow_build_threads = false;	/* E2: honour the `threads` reloption (the core's thread shortcut) instead of the worker build */
#endif
relopt_kind tp_relopt_kind;

void		_PG_init(void);

void
_PG_init(void)
{
	tp_shared_preload_init();	/* PostgreSQL 16: the shared-image registry needs shared_preload_libraries */
	tp_relopt_kind = add_reloption_kind();
	add_int_reloption(tp_relopt_kind, "cells", "Number of cells (0 = automatic: ~75 rows per cell, power of two, 16..8192)",
					  0, 0, 1 << 20, AccessExclusiveLock);
	add_int_reloption(tp_relopt_kind, "m", "Router graph degree (hnswlib M)", 32, 2, 128, AccessExclusiveLock);
	add_int_reloption(tp_relopt_kind, "ef_construction", "Router graph build beam (hnswlib efConstruction)",
					  200, 4, 4000, AccessExclusiveLock);
	add_int_reloption(tp_relopt_kind, "sample", "Rows sampled for the PCA head", 200000, 100, 100000000, AccessExclusiveLock);
	add_int_reloption(tp_relopt_kind, "kmeans_iters", "Lloyd iterations for the cells", 15, 1, 200, AccessExclusiveLock);
	add_int_reloption(tp_relopt_kind, "calibration_queries", "Corpus points used to calibrate the bound (0 = exact bound only)",
					  2000, 0, 200000, AccessExclusiveLock);
	add_int_reloption(tp_relopt_kind, "threads", "Build threads inside the core (PCA, k-means, codes + router, calibration); 1 = serial. "
					  "A measurement shortcut: the threads live only inside the C++ build phases.", 1, 1, 64, AccessExclusiveLock);
	add_int_reloption(tp_relopt_kind, "seed", "Seed of the PCA sample, the k-means init and the router's level draws (deterministic builds)",
					  0, 0, INT_MAX, AccessExclusiveLock);
	add_string_reloption(tp_relopt_kind, "levels", "Cumulative sketch widths, e.g. '32,128,256' ('' = by dimension)",
						 "", NULL, AccessExclusiveLock);

	DefineCustomIntVariable("trigpoint.probes", "Cells scanned per query (P).", NULL, &tp_probes, 32, 1, 1 << 20,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("trigpoint.hits", "Router hits per query (M).", NULL, &tp_hits, 200, 1, 100000,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("trigpoint.ef", "Router beam width (0 = trigpoint.hits).", NULL, &tp_ef, 0, 0, 100000,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("trigpoint.k", "Neighbours computed by the first search of an index scan; a query that asks for more widens the search (k doubled) until it is satisfied.",
							NULL, &tp_k, 10, 1, 10000, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("trigpoint.iterative_scan", "How a scan continues when the executor asks for more rows than its first search computed "
							 "(a selective WHERE above the index, or a LIMIT above trigpoint.k): off = the search is repeated with k doubled over the same cells; "
							 "relaxed_order = k and the cells opened double per batch, rows already fetched are not fetched again, a later batch may return a row "
							 "closer than one already returned; strict_order = as relaxed_order, rows that would come out of order are dropped.",
							 NULL, &tp_iterative, TP_ITER_RELAXED, tp_iterative_options, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("trigpoint.max_scan_tuples", "Heap rows an iterative scan fetches before it stops (the query then returns fewer rows than asked).",
							NULL, &tp_max_scan_tuples, 20000, 1, INT_MAX, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("trigpoint.max_probes", "Cells an iterative scan opens before it stops widening (0: every cell of the index).",
							NULL, &tp_max_probes, 0, 0, 1 << 20, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("trigpoint.compact_records", "VACUUM rewrites the index image when its log holds more than this many records, whatever their share (bounds a backend's replay at its first scan).",
							NULL, &tp_compact_records, 20000, 0, INT_MAX, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("trigpoint.max_shared_mb", "Cap on the index images pinned in shared memory across the cluster, in MB (0: no cap); a load beyond it fails with an ERROR.",
							NULL, &tp_max_shared_mb, 0, 0, INT_MAX, PGC_SUSET, GUC_UNIT_MB, NULL, NULL, NULL);
	DefineCustomEnumVariable("trigpoint.on_shared_cap", "What a backend does when loading an image into shared memory would exceed trigpoint.max_shared_mb: "
							 "error = the query fails (the default); private = the backend reads a private copy instead, one per backend, logged.",
							 NULL, &tp_on_shared_cap, TP_CAP_ERROR, tp_on_shared_cap_options, PGC_USERSET, 0, NULL, NULL, NULL);
#ifdef TRIGPOINT_TEST_HOOKS
	DefineCustomIntVariable("trigpoint.debug_load_sleep_ms", "Test hook: sleep this long inside a shared-image load, after the slot is claimed (test builds only).",
							NULL, &tp_debug_load_sleep_ms, 0, 0, 600000, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomBoolVariable("trigpoint.debug_load_fatal", "Test hook: raise FATAL inside a shared-image load, after the slot is claimed (test builds only).",
							 NULL, &tp_debug_load_fatal, false, PGC_USERSET, 0, NULL, NULL, NULL);
#endif
	DefineCustomIntVariable("trigpoint.recalibrate_queries", "Calibration queries a compaction runs over the live rows (0: keep the build's tables).",
							NULL, &tp_recalibrate_queries, 500, 0, 100000, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("trigpoint.recalibrate_probes", "Cells a compaction's calibration query opens, in the router's order (0: every cell, the exact top neighbours).",
							NULL, &tp_recalibrate_probes, 64, 0, INT_MAX, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("trigpoint.delta_flat_max", "Appended positions a backend routes by a flat scan before it links them into a delta graph (0: link every one at once).",
							NULL, &tp_delta_flat_max, 50000, 0, INT_MAX, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomRealVariable("trigpoint.delta", "Miss probability of the calibrated lower bound.", NULL, &tp_delta,
							 0.01, 0.0, 1.0, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomBoolVariable("trigpoint.exact", "Certificate mode: exact bound, every cell (100 % recall).", NULL,
							 &tp_exact, false, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomBoolVariable("trigpoint.merge_cells", "After trigpoint.seed_cells, cascade the remaining cells at once and fetch their survivors in one global lower-bound order (off: cell by cell).",
							 NULL, &tp_merge, true, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("trigpoint.seed_cells", "Cells scanned cell-by-cell to seed tau before the merged rest.", NULL, &tp_seed_cells, 1, 0, 1 << 20,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomRealVariable("trigpoint.compact_frac", "VACUUM rewrites the index image when inserts + tombstones since it exceed this fraction of its rows (and repacks cells above it).",
							 NULL, &tp_compact_frac, 0.2, 0.0001, 100.0, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomBoolVariable("trigpoint.compact_reclaim", "After a compaction, write the new image into the pages the retired generation freed and truncate the relation after its last live page (off: the file keeps the retired pages for reuse and never shrinks).",
							 NULL, &tp_compact_reclaim, true, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomBoolVariable("trigpoint.debug", "Report per-query statistics (candidates, heap fetches, core vs heap time) as a NOTICE.",
							 NULL, &tp_debug, false, PGC_USERSET, 0, NULL, NULL, NULL);
#ifdef TRIGPOINT_TEST_HOOKS
	DefineCustomBoolVariable("trigpoint.debug_skip_heap", "Diagnostic only: skip the heap fetches and rank candidates by their lower bound (wrong answers, isolates the core's time).",
							 NULL, &tp_debug_skip_heap, false, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomBoolVariable("trigpoint.debug_no_delta_graph", "Diagnostic only: route each scan with the image's router alone, the delta graph over the appended positions detached for the scan (their cells are still scanned when ranked). Off: today's behaviour.",
							 NULL, &tp_debug_no_delta_graph, false, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomBoolVariable("trigpoint.debug_centroid_router", "Ablation only: route each scan by centroid distance in the head space alone (no router graph, trigpoint.hits ignored); the first trigpoint.probes cells by centroid are scanned.",
							 NULL, &tp_debug_centroid_router, false, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomRealVariable("trigpoint.debug_ch", "Ablation only: 0 = the calibrated tail cosines (trigpoint.delta); a value in (0, 1] is used at every level instead (1 = the exact bound over the probed cells).",
							 NULL, &tp_debug_ch, 0.0, 0.0, 1.0, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomBoolVariable("trigpoint.allow_build_threads", "Honour the `threads` build option (the core's thread shortcut, measurement only); off: CREATE INDEX uses maintenance workers.",
							 NULL, &tp_allow_build_threads, false, PGC_USERSET, 0, NULL, NULL, NULL);
#endif
	DefineCustomBoolVariable("trigpoint.shared_image", "Search the index image from one shared-memory segment per index generation (off: every backend deserialises its own copy).",
							 NULL, &tp_shared, true, PGC_USERSET, 0, NULL, NULL, NULL);
	MarkGUCPrefixReserved("trigpoint");
}

PG_FUNCTION_INFO_V1(trigpoint_handler);
PG_FUNCTION_INFO_V1(trigpoint_version);
PG_FUNCTION_INFO_V1(trigpoint_info);

Datum
trigpoint_handler(PG_FUNCTION_ARGS)
{
	IndexAmRoutine *am = makeNode(IndexAmRoutine);

	am->amstrategies = 0;
	am->amsupport = 1;
	am->amoptsprocnum = 0;
	am->amcanorder = false;
	am->amcanorderbyop = true;
	am->amcanbackward = false;
	am->amcanunique = false;
	am->amcanmulticol = false;
	am->amoptionalkey = true;
	am->amsearcharray = false;
	am->amsearchnulls = false;
	am->amstorage = false;
	am->amclusterable = false;
	am->ampredlocks = false;
	am->amcanparallel = false;
#if PG_VERSION_NUM >= 170000
	am->amcanbuildparallel = false;
#endif
	am->amcaninclude = false;
	am->amusemaintenanceworkmem = false;
	am->amsummarizing = false;
	am->amparallelvacuumoptions = VACUUM_OPTION_NO_PARALLEL;
	am->amkeytype = InvalidOid;

	am->ambuild = tp_build;
	am->ambuildempty = tp_buildempty;
	am->aminsert = tp_insert;
#if PG_VERSION_NUM >= 170000
	am->aminsertcleanup = NULL;
#endif
	am->ambulkdelete = tp_bulkdelete;
	am->amvacuumcleanup = tp_vacuumcleanup;
	am->amcanreturn = NULL;
	am->amcostestimate = tp_costestimate;
	am->amoptions = tp_options;
	am->amproperty = NULL;
	am->ambuildphasename = NULL;
	am->amvalidate = tp_validate;
	am->amadjustmembers = NULL;
	am->ambeginscan = tp_beginscan;
	am->amrescan = tp_rescan;
	am->amgettuple = tp_gettuple;
	am->amgetbitmap = NULL;
	am->amendscan = tp_endscan;
	am->ammarkpos = NULL;
	am->amrestrpos = NULL;
	am->amestimateparallelscan = NULL;
	am->aminitparallelscan = NULL;
	am->amparallelrescan = NULL;

	PG_RETURN_POINTER(am);
}

/* trigpoint_version() -> text: extension version, core revision, distance kernel, build, server. */
/* H11: the metric of an index from its first column's operator family. By the family's catalog name, not a
 * name lookup: CREATE INDEX, REINDEX and VACUUM run under the restricted search_path (pg_catalog, pg_temp),
 * where "vector_cosine_ops" would never resolve. */
int
tp_index_metric(Relation index)
{
	HeapTuple	tup = SearchSysCache1(OPFAMILYOID, ObjectIdGetDatum(index->rd_opfamily[0]));
	int			metric = TP_METRIC_L2;

	if (HeapTupleIsValid(tup))
	{
		Form_pg_opfamily f = (Form_pg_opfamily) GETSTRUCT(tup);

		if (strcmp(NameStr(f->opfname), "vector_cosine_ops") == 0)
			metric = TP_METRIC_COSINE;
		ReleaseSysCache(tup);
	}
	return metric;
}

Datum
trigpoint_version(PG_FUNCTION_ARGS)
{
	char		buf[512];

	snprintf(buf, sizeof(buf), "trigpoint %s; core cascade.cpp@%s; kernel %s; %s; PostgreSQL %s%s",
			 TRIGPOINT_VERSION, tp_core_revision(), tp_core_kernel(), tp_core_build(), PG_VERSION,
#ifdef TRIGPOINT_TEST_HOOKS
			 "; test-hooks"
#else
			 ""
#endif
		);
	PG_RETURN_TEXT_P(cstring_to_text(buf));
}

/* trigpoint_info(regclass) -> text: what the meta page records about an index (sizes, timings). */
Datum
trigpoint_info(PG_FUNCTION_ARGS)
{
	Oid			relid = PG_GETARG_OID(0);
	Relation	index = tp_open_index_checked(relid, AccessShareLock, false);
	TrigpointMetaPageData meta;
	StringInfoData s;
	int			l;

	tp_read_meta(index, &meta);
	initStringInfo(&s);
	appendStringInfo(&s, "n=%d D=%d metric=%s cells=%d levels=", meta.n, meta.D, meta.metric == TP_METRIC_COSINE ? "cosine" : "l2", meta.C);
	for (l = 0; l < meta.L; l++)
		appendStringInfo(&s, "%s%d", l ? "," : "", meta.lv[l]);
	appendStringInfo(&s, " m=%d ef_construction=%d image_pages=%u rel_pages=%u image_bytes=" UINT64_FORMAT " core_bytes=" UINT64_FORMAT
					 " generation=%u n_image=%d log_inserts=" INT64_FORMAT " log_deletes=" INT64_FORMAT " compactions=%d"
					 " var_captured=%.4f ch_default=", meta.M, meta.efC, meta.nchunks, RelationGetNumberOfBlocks(index), meta.blob_bytes,
					 meta.core_bytes, meta.generation, meta.n_image, meta.log_inserts, meta.log_deletes, meta.compactions, meta.var_captured);
	for (l = 0; l < meta.L; l++)
		appendStringInfo(&s, "%s%.3f", l ? "," : "", meta.ch_default[l]);
	appendStringInfo(&s, " build_s: scan=%.1f fit=%.1f head=%.1f kmeans=%.1f index=%.1f (codes=%.1f graph=%.1f) calib=%.1f write=%.1f total=%.1f participants=%d",
					 meta.t_scan, meta.t_fit, meta.t_head, meta.t_kmeans, meta.t_index, meta.t_codes, meta.t_graph, meta.t_calib, meta.t_write,
					 meta.t_scan + meta.t_fit + meta.t_head + meta.t_kmeans + meta.t_index + meta.t_calib + meta.t_write, meta.participants);
	relation_close(index, AccessShareLock);
	PG_RETURN_TEXT_P(cstring_to_text(s.data));
}

Relation
tp_open_index_checked(Oid relid, LOCKMODE lockmode, bool need_owner)
{
	Relation	index = relation_open(relid, lockmode);
	Oid			heapid;

	/* checked after the lock is held, so the relation cannot change under us (TOCTOU) */
	if (index->rd_rel->relkind != RELKIND_INDEX || index->rd_rel->relam != get_index_am_oid("trigpoint", false))
		ereport(ERROR, (errcode(ERRCODE_WRONG_OBJECT_TYPE), errmsg("\"%s\" is not a trigpoint index", RelationGetRelationName(index))));
	heapid = IndexGetRelation(relid, false);
	if (need_owner)
	{
		if (!object_ownercheck(RelationRelationId, heapid, GetUserId()))
			aclcheck_error(ACLCHECK_NOT_OWNER, get_relkind_objtype(get_rel_relkind(heapid)), get_rel_name(heapid));
	}
	else
	{
		AclResult	acl = pg_class_aclcheck(heapid, GetUserId(), ACL_SELECT);

		if (acl != ACLCHECK_OK)
			aclcheck_error(acl, get_relkind_objtype(get_rel_relkind(heapid)), get_rel_name(heapid));
	}
	return index;
}

bool
tp_validate(Oid opclassoid)
{
	return true;
}

bytea *
tp_options(Datum reloptions, bool validate)
{
	static const relopt_parse_elt tab[] = {
		{"cells", RELOPT_TYPE_INT, offsetof(TrigpointOptions, cells)},
		{"m", RELOPT_TYPE_INT, offsetof(TrigpointOptions, m)},
		{"ef_construction", RELOPT_TYPE_INT, offsetof(TrigpointOptions, ef_construction)},
		{"sample", RELOPT_TYPE_INT, offsetof(TrigpointOptions, sample)},
		{"kmeans_iters", RELOPT_TYPE_INT, offsetof(TrigpointOptions, kmeans_iters)},
		{"calibration_queries", RELOPT_TYPE_INT, offsetof(TrigpointOptions, calibration_queries)},
		{"threads", RELOPT_TYPE_INT, offsetof(TrigpointOptions, threads)},
		{"seed", RELOPT_TYPE_INT, offsetof(TrigpointOptions, seed)},
		{"levels", RELOPT_TYPE_STRING, offsetof(TrigpointOptions, levels_offset)},
	};

	return (bytea *) build_reloptions(reloptions, validate, tp_relopt_kind, sizeof(TrigpointOptions), tab, lengthof(tab));
}
