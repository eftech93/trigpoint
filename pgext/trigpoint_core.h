/* C surface of the Trigpoint core for the PostgreSQL module.
 *
 * trigpoint.c / tp_build.c / tp_scan.c are C (PostgreSQL's headers are C); the core is C++17
 * (csrc/cascade.cpp plus the shim trigpoint_core.cpp), compiled into libtrigpoint_core.a and linked
 * into trigpoint.so by PGXS. Rules that keep the two worlds apart:
 *   - nothing on the C++ side includes PostgreSQL headers or calls back into the server; the only
 *     callback is tp_abort_fn, which the module implements as a read of its interrupt flags;
 *   - no C++ exception crosses into the backend: every entry point catches and returns an error
 *     code / NULL, with the message in tp_last_error();
 *   - the core's memory is malloc'd here and released by the explicit *_free calls (the module calls
 *     them from its relcache-invalidation callback, scan end and error paths), never palloc'd.
 */
#ifndef TRIGPOINT_CORE_H
#define TRIGPOINT_CORE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

const char *tp_core_revision(void);   /* sha256 prefix of csrc/cascade.cpp the library was built from */
const char *tp_core_kernel(void);     /* "avx2+fma" | "neon+dotprod" | "neon" | "scalar" (cascade_kernel()) */
const char *tp_core_build(void);      /* compiler version and architecture flags */
const char *tp_last_error(void);      /* message of the last failure in this thread, "" if none */

typedef int (*tp_abort_fn)(void *ctx);   /* return non-zero to cancel a build cooperatively */

/* ---- build: collect the heap's vectors, then build the index ----------------------------------- */
void       *tp_builder_new(int D, int64_t expected_rows);       /* NULL on allocation failure */
int         tp_builder_add(void *b, const float *v, uint64_t tid); /* 0, or -1 on allocation failure */
int64_t     tp_builder_count(const void *b);
const uint64_t *tp_builder_tids(const void *b);                /* valid until tp_builder_free */
/* prm = {cells, L, lv0, lv1, lv2, lv3, M, ef_construction, sample, kmeans_iters, kmeans_train,
 * calibration_queries, power_iters, seed, threads}; stats (6 doubles, may be NULL) = {t_fit, t_head,
 * t_kmeans, t_index, t_calib, var_captured}. The vectors are released on success; the tids stay. */
void       *tp_builder_build(void *b, const int *prm, double *stats, tp_abort_fn abort_fn, void *ctx);
void       *tp_index_build_trivial(int D, int L, const int *lv, int M, int efC);   /* H14: the model of an index over no rows; NULL + tp_last_error() */
void        tp_builder_free(void *b);

/* ---- index image ------------------------------------------------------------------------------- */
int         tp_index_serialize(void *ix, void **buf, size_t *len); /* malloc'd; tp_free_buffer */
void        tp_free_buffer(void *buf);
void       *tp_index_deserialize(const void *buf, size_t len);     /* NULL + tp_last_error() */
void        tp_index_free(void *ix);
/* E1 shared image: the flat, position-independent form of an owned index (buf NULL -> bytes needed;
 * every cell fully repacked first), and an index that reads such an image in place (the buffer must
 * outlive it; may be mapped read-only by several processes) keeping inserts and tombstones private;
 * tp_index_private_bytes = that private memory. A flat-opened index cannot be serialised or compacted. */
long long   tp_index_flatten(void *ix, void *buf, size_t cap);
void       *tp_index_open_flat(const void *buf, size_t len);         /* NULL + tp_last_error() */
long long   tp_index_private_bytes(void *ix);
int         tp_flat_check(const void *buf, size_t len, int deep);  /* H3: 0 ok, -1 + tp_last_error(); deep = O(n) contents */
/* E2: the router is a FlatGraph (csrc/flat_graph.h) built by the extension (pgext/tp_graph.c). A frozen index
 * reads the image's graph; the positions appended since the image are routed through a private delta graph
 * the extension builds and hands over here (id i = position nb + i; re-set after every growth). The
 * compaction copies the base graph (tp_index_base_graph), inserts the delta, and re-flattens with it. */
void        tp_index_set_delta_graph(void *ix, const void *graph);
const void *tp_index_base_graph(void *ix);
int         tp_index_router_params(void *ix, int *out3, const float **step);   /* {d3, M, ef_construction}, the steps */
int         tp_index_router_mode(void *ix);                                   /* 0: the M best hits mark cells; 1: all ef results; 2: the whole visited set */

/* ---- E2: range-based build phases for the parallel CREATE INDEX (tp_build.c). Pure array work over
 * [lo, hi) of caller-provided (DSM) arrays; negative return = allocation failure (tp_last_error). ---- */
int         tp_sample(uint64_t seed, int n, int m, int *out);                   /* m distinct sorted indices of n; returns the count */
void        tp_mean(const float *x, const int *idx, int m, int D, float *mean);
int         tp_cov_partial(const float *x, const int *idx, int lo, int hi, int D, const float *mean, double *cov);  /* upper triangle added */
float       tp_pca_solve(double *cov, int m, int D, int d3, int iters, uint64_t seed, float *proj);  /* returns the variance share */
/* the same solve split for the participants: prepare (normalise, trace), rank r, init Q (r x D), per-iteration
 * Y[j] = cov Q[j] over columns [j_lo, j_hi), orthonormalise the swapped Q, finish (Rayleigh-Ritz -> proj, variance share) */
double      tp_pca_prepare(double *cov, int m, int D);
int         tp_pca_rank(int D, int d3);
int         tp_pca_init(double *Q, int D, int r, uint64_t seed);
void        tp_pca_mul_partial(const double *cov, int D, const double *Q, int j_lo, int j_hi, double *Y);
int         tp_orthonormalise(double *Q, int D, int r);
float       tp_pca_finish(const double *Q, const double *Y, int D, int r, int d3, double trace, float *proj);
int         tp_project_rows(const float *x, int lo, int hi, int D, int d3, const float *proj, const float *mean, float *H);
void        tp_minmax_partial(const float *H, int lo, int hi, int d3, float *mn, float *mx);
void        tp_lo_step(const float *mn, const float *mx, int d3, float *lo, float *step);
int         tp_kmeans_init(const float *H, const int *train_idx, int T, int d3, int C, uint64_t seed, float *cent);
int         tp_kmeans_assign_partial(const float *H, const int *idx, int lo, int hi, int d3, const float *cent, int C, int *labels, double *sum, int *cnt);
int         tp_kmeans_update(const float *H, const int *train_idx, int T, int d3, int C, const double *sum, const int *cnt, uint64_t seed, int iter, float *cent);
void        tp_encode_row(const float *x, const float *h, int D, int L, const int *lv, const float *lo, const float *step, const float *mean, int8_t *codes, float *err_out, float *tail_out);
long long   tp_flat_layout(void *hdr, int n, int D, int C, int L, const int *lv, int M, int efC, int npos_of, const uint64_t *cal_cap, const int *cell_cnt, uint64_t graph_bytes);
int         tp_calibrate_query(void *frozen, const float *x, int D, const float *H, const int *order, int t, float *out, int *cnt);  /* order: id -> row (NULL = identity); 1 = usable, 0 = too few neighbours */
float       tp_ch_from_sorted(const float *g, int n, double delta);
int         tp_route_hits(void *ix, const float *q, int M, int ef, int *ids, float *scores);   /* the router's top-M hits as ids + scores */
long long   tp_index_reflatten(void *ix, const void *graph, void *buf, size_t cap);
/* H13 (compaction that reclaims dead rows): newpos[pos] = rank among the live positions or -1 (returns the live
 * count); the id a position carries; the position of an id (-1 unknown); the renumbered image over the live
 * positions whose router is `graph` (built over the new positions): id = position, no tombstones. */
int         tp_index_live_map(void *ix, int32_t *newpos);
int         tp_index_pos_id(void *ix, int pos);
int         tp_index_id_pos(void *ix, int id);
long long   tp_index_reflatten_live(void *ix, const void *graph, const int32_t *newpos, int n_new, const uint64_t *cal_cap, void *buf, size_t cap);
/* H16: re-calibration at compaction over a flat-opened image: the query's exact sketch, and the critical cosine of a
 * (query, row) pair at the k-th distance tau2 (calibrate_index's formula); returns L, or -1 for an unknown id. */
void        tp_query_sketch(void *ix, const float *x, float *head, float *tail4);
int         tp_calibrate_pair(void *ix, const float *head, const float *tail4, int id, float tau2, float *outL);
float       tp_sq_dist(const float *a, const float *b, int D);
int         tp_index_info(void *ix, long long *out14);
int         tp_index_ch(void *ix, double delta, float *out4);      /* calibrated tail cosines */
float       tp_index_var_captured(void *ix);
/* incremental maintenance: encode a vector (returns its cell; codes = d3 int8, err/tail = L floats),
 * apply an encoded insert (returns the position, < 0 on error), tombstone ids, compact cells whose
 * overflow + tombstones exceed frac of their main block (returns cells compacted), counts = {total,
 * inserted, deleted} */
int         tp_index_encode(void *ix, const float *x, int8_t *codes, float *err, float *tail);
int         tp_index_insert(void *ix, int id, int cell, const int8_t *codes, const float *err, const float *tail);
void        tp_index_delete(void *ix, const int *ids, int count);
int         tp_index_is_deleted(void *ix, int id);    /* 1 when tombstoned (or unknown) */
int         tp_index_compact(void *ix, float frac);
void        tp_index_counts(void *ix, long long *out3);

/* ---- two-phase scan: route once, then one cell per call; the caller computes exact distances ---- */
void       *tp_scan_begin(void *ix, const float *q, int M, int P, int ef, const float *ch4);
int         tp_scan_cell(void *sc, float tau2);   /* survivors of the next cell, -1 when done, -2 error */
int         tp_scan_rest(void *sc, float tau2);   /* survivors of all remaining cells, merged by bound */
int         tp_scan_some(void *sc, float tau2, int ncells);  /* the next ncells cells at most, merged by bound (tp_scan_rest = every one) */
int         tp_scan_restart(void *sc, int P);     /* iterative scan: rewind the cell cursor and cover the first P cells of the order (the un-hit rest ranked by centroid once); cells covered, -2 error */
void        tp_scan_take(void *sc, int *ids, float *lb2);
void        tp_scan_stats(void *sc, long long *out5); /* dim-ops, hits, cells hit, cells scanned, cells ranked */
void        tp_scan_end(void *sc);
float       tp_dist_i8(const int8_t *a, const int8_t *b, const float *w2, int d);   /* sum_i w2[i] (a[i]-b[i])^2: the router's distance */

#ifdef __cplusplus
}
#endif
#endif
