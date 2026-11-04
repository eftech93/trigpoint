// C shim around csrc/cascade.cpp (see trigpoint_core.h). Compiled with the same flags as the core.
// Every entry point catches: hnswlib throws std::runtime_error, the standard library bad_alloc;
// the module sees NULL / a negative code and reads tp_last_error().
#include "trigpoint_core.h"

#include <cstring>
#include <exception>
#include <new>
#include <string>
#include <vector>

extern "C" {  // the core's C ABI (csrc/cascade.cpp)
const char* cascade_kernel();
void* cascade_build_vectors(const float* x, int n, int D, const int* prm, double* stats_out,
                            tp_abort_fn abort_fn, void* abort_ctx, char* err, int errlen);
void cascade_free(void* p);
int cascade_serialize(void* p, void** out, size_t* out_len);
void cascade_free_buffer(void* buf);
void* cascade_deserialize(const void* buf, size_t len, char* err, int errlen);
long long cascade_flatten(void* p, void* buf, size_t cap);
long long cascade_reflatten(void* p, const void* graph, void* buf, size_t cap);
int cascade_live_map(void* p, int32_t* newpos);
int cascade_pos_id(void* p, int pos);
int cascade_id_pos(void* p, int id);
long long cascade_reflatten_live(void* p, const void* graph, const int32_t* newpos, int n_new, const uint64_t* cal_cap, void* buf, size_t cap);
void cascade_query_sketch(void* p, const float* x, float* head, float* tail4);
int cascade_calibrate_pair(void* p, const float* head, const float* tail4, int id, float tau2, float* out);
float cascade_sq_dist(const float* a, const float* b, int D);
void cascade_set_delta_graph(void* p, const void* graph);
int cascade_get_router_mode(void* p);
const void* cascade_base_graph(void* p);
int cascade_router_params(void* p, int* out3, const float** step);
int cascade_sample(uint64_t seed, int n, int m, int* out);
void cascade_mean(const float* x, const int* idx, int m, int D, float* mean);
void cascade_cov_partial(const float* x, const int* idx, int lo, int hi, int D, const float* mean, double* cov);
float cascade_pca_solve(double* cov, int m, int D, int d3, int iters, uint64_t seed, float* proj);
double cascade_pca_prepare(double* cov, int m, int D);
int cascade_pca_rank(int D, int d3);
void cascade_pca_init(double* Q, int D, int r, uint64_t seed);
void cascade_pca_mul_partial(const double* cov, int D, const double* Q, int j_lo, int j_hi, double* Y);
void cascade_orthonormalise(double* Q, int D, int r);
float cascade_pca_finish(const double* Q, const double* Y, int D, int r, int d3, double trace, float* proj);
void cascade_project_rows(const float* x, int lo, int hi, int D, int d3, const float* proj, const float* mean, float* H);
void cascade_minmax_partial(const float* H, int lo, int hi, int d3, float* mn, float* mx);
void cascade_lo_step(const float* mn, const float* mx, int d3, float* lo, float* step);
void cascade_kmeans_init(const float* H, const int* train_idx, int T, int d3, int C, uint64_t seed, float* cent);
void cascade_kmeans_assign_partial(const float* H, const int* idx, int lo, int hi, int d3, const float* cent, int C, int* labels, double* sum, int* cnt);
void cascade_kmeans_update(const float* H, const int* train_idx, int T, int d3, int C, const double* sum, const int* cnt, uint64_t seed, int iter, float* cent);
void cascade_encode_row(const float* x, const float* h, int D, int L, const int* lv, const float* lo, const float* step, const float* mean, int8_t* codes, float* err_out, float* tail_out);
long long cascade_flat_layout(void* hdr, int n, int D, int C, int L, const int* lv, int M, int efC, int npos_of, const uint64_t* cal_cap, const int* cell_cnt, uint64_t graph_bytes);
int cascade_calibrate_query(void* frozen, const float* x, int D, const float* H, const int* order, int t, float* out, int* cnt);
float cascade_ch_from_sorted(const float* g, int n, double delta);
int cascade_route_hits(void* p, const float* q, int M, int ef, int* ids_out, float* scores_out);
void* cascade_open_flat(const void* buf, size_t len, char* err, int errlen);
void* cascade_build_trivial(int D, int L, const int* lv, int M, int efC, char* err, int errlen);
int cascade_check_flat(const void* buf, size_t len, int deep, char* err, int errlen);
long long cascade_private_bytes(void* p);
int cascade_info(void* p, long long* out);
int cascade_ch_for_delta(void* p, double delta, float* out);
float cascade_var_captured(void* p);
void* cascade_scan_begin(void* p, const float* q, int M, int P, int ef, const float* ch);
int cascade_scan_cell(void* scp, float tau2);
int cascade_scan_rest(void* scp, float tau2);
int cascade_scan_some(void* scp, float tau2, int ncells);
int cascade_scan_restart(void* scp, int P);
void cascade_scan_take(void* scp, int* ids, float* lb2);
void cascade_scan_stats(void* scp, long long* out);
void cascade_scan_end(void* scp);
float cascade_sq_dist(const float* a, const float* b, int D);
float cascade_dist_i8(const int8_t* a, const int8_t* b, const float* w2, int d);
int cascade_encode(void* p, const float* x, int8_t* codes_out, float* err_out, float* tail_out);
int cascade_insert_codes(void* p, int id, int cell, const int8_t* codes, const float* err, const float* tail);
void cascade_delete(void* p, const int* ids, int count);
int cascade_is_deleted(void* p, int id);
int cascade_compact(void* p, float frac);
long long cascade_counts(void* p, long long* out);
}

namespace {
thread_local std::string g_last_error;
void fail(const char* what) { g_last_error = what; }
void fail(const std::exception& e) { g_last_error = e.what(); }

struct Builder {
    int D;
    std::vector<float> x;
    std::vector<uint64_t> tids;
};
}  // namespace

extern "C" {

const char* tp_core_revision(void) { return TRIGPOINT_CORE_REV; }
const char* tp_core_kernel(void) { return cascade_kernel(); }
const char* tp_core_build(void) {
#if defined(__clang__)
    return "clang " __clang_version__ " " TRIGPOINT_ARCHFLAGS;
#else
    return "g++ " __VERSION__ " " TRIGPOINT_ARCHFLAGS;
#endif
}
const char* tp_last_error(void) { return g_last_error.c_str(); }

void* tp_builder_new(int D, int64_t expected_rows) {
    try {
        Builder* b = new Builder();
        b->D = D;
        if (expected_rows > 0) { b->x.reserve((size_t)expected_rows * (size_t)D); b->tids.reserve((size_t)expected_rows); }
        return b;
    } catch (const std::exception& e) { fail(e); return nullptr; }
}

int tp_builder_add(void* bp, const float* v, uint64_t tid) {
    Builder* b = (Builder*)bp;
    try { b->x.insert(b->x.end(), v, v + b->D); b->tids.push_back(tid); return 0; }
    catch (const std::exception& e) { fail(e); return -1; }
}

int64_t tp_builder_count(const void* bp) { return (int64_t)((const Builder*)bp)->tids.size(); }
const uint64_t* tp_builder_tids(const void* bp) { return ((const Builder*)bp)->tids.data(); }

void* tp_builder_build(void* bp, const int* prm, double* stats, tp_abort_fn abort_fn, void* ctx) {
    Builder* b = (Builder*)bp;
    char err[256] = {0};
    try {
        void* ix = cascade_build_vectors(b->x.data(), (int)b->tids.size(), b->D, prm, stats, abort_fn, ctx, err, sizeof err);
        if (!ix) { fail(err[0] ? err : "build failed"); return nullptr; }
        std::vector<float>().swap(b->x);  // the index is sketch-only; release the vectors now
        return ix;
    } catch (const std::exception& e) { fail(e); return nullptr; }
}

void tp_builder_free(void* bp) { delete (Builder*)bp; }
void* tp_index_build_trivial(int D, int L, const int* lv, int M, int efC) {
    char err[256] = {0};
    try { void* ix = cascade_build_trivial(D, L, lv, M, efC, err, sizeof err); if (!ix) fail(err[0] ? err : "trivial build failed"); return ix; }
    catch (const std::exception& e) { fail(e); return nullptr; }
}

int tp_index_serialize(void* ix, void** buf, size_t* len) {
    try { int rc = cascade_serialize(ix, buf, len); if (rc) fail("serialisation failed"); return rc; }
    catch (const std::exception& e) { fail(e); return -9; }
}
void tp_free_buffer(void* buf) { cascade_free_buffer(buf); }

void* tp_index_deserialize(const void* buf, size_t len) {
    char err[256] = {0};
    try { void* ix = cascade_deserialize(buf, len, err, sizeof err); if (!ix) fail(err[0] ? err : "deserialisation failed"); return ix; }
    catch (const std::exception& e) { fail(e); return nullptr; }
}
void tp_index_free(void* ix) { if (ix) cascade_free(ix); }
long long tp_index_flatten(void* ix, void* buf, size_t cap) {
    try { long long r = cascade_flatten(ix, buf, cap); if (r < 0) fail("index cannot be flattened (frozen or reduced router)"); return r; }
    catch (const std::exception& e) { fail(e); return -9; }
}
void* tp_index_open_flat(const void* buf, size_t len) {
    char err[256] = {0};
    try { void* ix = cascade_open_flat(buf, len, err, sizeof err); if (!ix) fail(err[0] ? err : "cannot open the flat image"); return ix; }
    catch (const std::exception& e) { fail(e); return nullptr; }
}
long long tp_index_private_bytes(void* ix) { return cascade_private_bytes(ix); }
int tp_flat_check(const void* buf, size_t len, int deep) {
    char err[256] = {0};
    try { int rc = cascade_check_flat(buf, len, deep, err, sizeof err); if (rc < 0) fail(err[0] ? err : "corrupt flat image"); return rc; }
    catch (const std::exception& e) { fail(e); return -9; }
}
long long tp_index_reflatten(void* ix, const void* graph, void* buf, size_t cap) {
    try { long long r = cascade_reflatten(ix, graph, buf, cap); if (r < 0) fail("index cannot be re-flattened (not frozen, or the graph does not cover every position)"); return r; }
    catch (const std::exception& e) { fail(e); return -9; }
}
int tp_index_live_map(void* ix, int32_t* newpos) { return cascade_live_map(ix, newpos); }
int tp_index_pos_id(void* ix, int pos) { return cascade_pos_id(ix, pos); }
int tp_index_id_pos(void* ix, int id) { return cascade_id_pos(ix, id); }
void tp_query_sketch(void* ix, const float* x, float* head, float* tail4) { cascade_query_sketch(ix, x, head, tail4); }
int tp_calibrate_pair(void* ix, const float* head, const float* tail4, int id, float tau2, float* outL) { return cascade_calibrate_pair(ix, head, tail4, id, tau2, outL); }
float tp_sq_dist(const float* a, const float* b, int D) { return cascade_sq_dist(a, b, D); }
long long tp_index_reflatten_live(void* ix, const void* graph, const int32_t* newpos, int n_new, const uint64_t* cal_cap, void* buf, size_t cap) {
    try { long long r = cascade_reflatten_live(ix, graph, newpos, n_new, cal_cap, buf, cap); if (r < 0) fail("renumbered image cannot be written (frozen index, graph or map)"); return r; }
    catch (const std::exception& e) { fail(e); return -9; }
}
void tp_index_set_delta_graph(void* ix, const void* graph) { cascade_set_delta_graph(ix, graph); }
const void* tp_index_base_graph(void* ix) { return cascade_base_graph(ix); }
int tp_index_router_params(void* ix, int* out3, const float** step) { return cascade_router_params(ix, out3, step); }
int tp_index_router_mode(void* ix) { return cascade_get_router_mode(ix); }

// range-based build phases (parallel CREATE INDEX); pure array work, exceptions only from allocation
int tp_sample(uint64_t seed, int n, int m, int* out) { try { return cascade_sample(seed, n, m, out); } catch (const std::exception& e) { fail(e); return -1; } }
void tp_mean(const float* x, const int* idx, int m, int D, float* mean) { cascade_mean(x, idx, m, D, mean); }
int tp_cov_partial(const float* x, const int* idx, int lo, int hi, int D, const float* mean, double* cov) { try { cascade_cov_partial(x, idx, lo, hi, D, mean, cov); return 0; } catch (const std::exception& e) { fail(e); return -1; } }
float tp_pca_solve(double* cov, int m, int D, int d3, int iters, uint64_t seed, float* proj) { try { return cascade_pca_solve(cov, m, D, d3, iters, seed, proj); } catch (const std::exception& e) { fail(e); return -1.f; } }
double tp_pca_prepare(double* cov, int m, int D) { return cascade_pca_prepare(cov, m, D); }
int tp_pca_rank(int D, int d3) { return cascade_pca_rank(D, d3); }
int tp_pca_init(double* Q, int D, int r, uint64_t seed) { try { cascade_pca_init(Q, D, r, seed); return 0; } catch (const std::exception& e) { fail(e); return -1; } }
void tp_pca_mul_partial(const double* cov, int D, const double* Q, int j_lo, int j_hi, double* Y) { cascade_pca_mul_partial(cov, D, Q, j_lo, j_hi, Y); }
int tp_orthonormalise(double* Q, int D, int r) { try { cascade_orthonormalise(Q, D, r); return 0; } catch (const std::exception& e) { fail(e); return -1; } }
float tp_pca_finish(const double* Q, const double* Y, int D, int r, int d3, double trace, float* proj) { try { return cascade_pca_finish(Q, Y, D, r, d3, trace, proj); } catch (const std::exception& e) { fail(e); return -1.f; } }
int tp_project_rows(const float* x, int lo, int hi, int D, int d3, const float* proj, const float* mean, float* H) { try { cascade_project_rows(x, lo, hi, D, d3, proj, mean, H); return 0; } catch (const std::exception& e) { fail(e); return -1; } }
void tp_minmax_partial(const float* H, int lo, int hi, int d3, float* mn, float* mx) { cascade_minmax_partial(H, lo, hi, d3, mn, mx); }
void tp_lo_step(const float* mn, const float* mx, int d3, float* lo, float* step) { cascade_lo_step(mn, mx, d3, lo, step); }
int tp_kmeans_init(const float* H, const int* train_idx, int T, int d3, int C, uint64_t seed, float* cent) { try { cascade_kmeans_init(H, train_idx, T, d3, C, seed, cent); return 0; } catch (const std::exception& e) { fail(e); return -1; } }
int tp_kmeans_assign_partial(const float* H, const int* idx, int lo, int hi, int d3, const float* cent, int C, int* labels, double* sum, int* cnt) { try { cascade_kmeans_assign_partial(H, idx, lo, hi, d3, cent, C, labels, sum, cnt); return 0; } catch (const std::exception& e) { fail(e); return -1; } }
int tp_kmeans_update(const float* H, const int* train_idx, int T, int d3, int C, const double* sum, const int* cnt, uint64_t seed, int iter, float* cent) { try { cascade_kmeans_update(H, train_idx, T, d3, C, sum, cnt, seed, iter, cent); return 0; } catch (const std::exception& e) { fail(e); return -1; } }
void tp_encode_row(const float* x, const float* h, int D, int L, const int* lv, const float* lo, const float* step, const float* mean, int8_t* codes, float* err_out, float* tail_out) { cascade_encode_row(x, h, D, L, lv, lo, step, mean, codes, err_out, tail_out); }
long long tp_flat_layout(void* hdr, int n, int D, int C, int L, const int* lv, int M, int efC, int npos_of, const uint64_t* cal_cap, const int* cell_cnt, uint64_t graph_bytes) { return cascade_flat_layout(hdr, n, D, C, L, lv, M, efC, npos_of, cal_cap, cell_cnt, graph_bytes); }
int tp_calibrate_query(void* frozen, const float* x, int D, const float* H, const int* order, int t, float* out, int* cnt) { try { return cascade_calibrate_query(frozen, x, D, H, order, t, out, cnt); } catch (const std::exception& e) { fail(e); return -1; } }
float tp_ch_from_sorted(const float* g, int n, double delta) { return cascade_ch_from_sorted(g, n, delta); }
int tp_route_hits(void* ix, const float* q, int M, int ef, int* ids, float* scores) { try { return cascade_route_hits(ix, q, M, ef, ids, scores); } catch (const std::exception& e) { fail(e); return -1; } }
int tp_index_info(void* ix, long long* out) { return cascade_info(ix, out); }
int tp_index_ch(void* ix, double delta, float* out) { return cascade_ch_for_delta(ix, delta, out); }
float tp_index_var_captured(void* ix) { return cascade_var_captured(ix); }

void* tp_scan_begin(void* ix, const float* q, int M, int P, int ef, const float* ch) {
    try { return cascade_scan_begin(ix, q, M, P, ef, ch); }
    catch (const std::exception& e) { fail(e); return nullptr; }
}
int tp_scan_cell(void* sc, float tau2) {
    try { return cascade_scan_cell(sc, tau2); }
    catch (const std::exception& e) { fail(e); return -2; }
}
int tp_scan_rest(void* sc, float tau2) {
    try { return cascade_scan_rest(sc, tau2); }
    catch (const std::exception& e) { fail(e); return -2; }
}
int tp_scan_some(void* sc, float tau2, int ncells) {
    try { return cascade_scan_some(sc, tau2, ncells); }
    catch (const std::exception& e) { fail(e); return -2; }
}
int tp_scan_restart(void* sc, int P) {
    try { return cascade_scan_restart(sc, P); }
    catch (const std::exception& e) { fail(e); return -2; }
}
void tp_scan_take(void* sc, int* ids, float* lb2) { cascade_scan_take(sc, ids, lb2); }
void tp_scan_stats(void* sc, long long* out) { cascade_scan_stats(sc, out); }
void tp_scan_end(void* sc) { if (sc) cascade_scan_end(sc); }
float tp_dist_i8(const int8_t* a, const int8_t* b, const float* w2, int d) { return cascade_dist_i8(a, b, w2, d); }

int tp_index_encode(void* ix, const float* x, int8_t* codes, float* err, float* tail) {
    try { return cascade_encode(ix, x, codes, err, tail); }
    catch (const std::exception& e) { fail(e); return -9; }
}
int tp_index_insert(void* ix, int id, int cell, const int8_t* codes, const float* err, const float* tail) {
    try { int r = cascade_insert_codes(ix, id, cell, codes, err, tail); if (r < 0) fail(r == -2 ? "id already in use" : "bad cell / id"); return r; }
    catch (const std::exception& e) { fail(e); return -9; }
}
void tp_index_delete(void* ix, const int* ids, int count) { cascade_delete(ix, ids, count); }
int tp_index_is_deleted(void* ix, int id) { return cascade_is_deleted(ix, id); }
int tp_index_compact(void* ix, float frac) {
    try { return cascade_compact(ix, frac); }
    catch (const std::exception& e) { fail(e); return -9; }
}
void tp_index_counts(void* ix, long long* out3) { cascade_counts(ix, out3); }

}  // extern "C"
