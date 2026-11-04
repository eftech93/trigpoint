// Native single-node search: sketch-routed cells + multi-resolution int8 cascade,
// with inserts (per-cell overflow blocks) and deletes (tombstones).
//
// Layout (built once from Python, see src/vsroute/native.py):
//   * points sorted by cell; per cell the int8 head codes are stored level-blocked
//     (all points' dims [0,d1), then all points' dims [d1,d2), ...), so a level-1
//     scan of a cell is one contiguous read of cnt*d1 bytes;
//   * inserted points go to a per-cell overflow block (row-major codes), scanned
//     after the main block; deleted points are tombstoned and skipped at the
//     exact phase (and marked deleted in the router);
//   * per point and level: the quantisation error norm and the PCA-tail norm;
//   * full float32 vectors in the original space, read only for final candidates;
//   * an HNSW over the int8 heads (the router), mixed float/int8 NEON / AVX2 kernels.
//
// Search (one query, single thread):
//   1. project q onto the first d3 PCA axes (d3*D flops) and derive the per-level
//      tail norms from ||q - mean||;
//   2. router: top-M sketch hits -> cells ranked by best hit, then by centroid
//      (router_mode 1 / 2: cells of all ef results / of the beam's whole visited set);
//   3. for the first P cells, level cascade: lb_h <= tau keeps a point alive,
//      lb_h^2 = max(||q_h - x^_h|| - err_h, 0)^2 + max(a^2 + b^2 - 2 c_h a b, 0);
//      c_h = 1 is the exact lower bound, c_h < 1 the conformally calibrated one;
//   4. survivors get exact distances in bound order, tightening tau.
// With P = C and c_h = 1 the search is exact (a certificate on every query).

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <queue>
#include <chrono>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <signal.h>
#include <pthread.h>

#include "hnswlib/hnswlib.h"
#include "flat_graph.h"
#include "flat_image.h"
#include "flat_check.h"
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace {

// x86: AVX2 + FMA kernels (the counterpart of the NEON kernels; -march=native enables them on
// any Haswell-or-later CPU). Without them the compiler keeps the float reductions scalar.
#if defined(__AVX2__) && defined(__FMA__)
inline float hsum256(__m256 v) {
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}
inline __m256 load_i8_as_ps(const int8_t* p) {  // 8 int8 -> 8 float
    return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)p)));
}
#endif

inline float sq_dist_full(const float* a, const float* b, int D) {
#if defined(__ARM_NEON)
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
    int i = 0;
    for (; i + 16 <= D; i += 16) {
        float32x4_t t0 = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        float32x4_t t1 = vsubq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        float32x4_t t2 = vsubq_f32(vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        float32x4_t t3 = vsubq_f32(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
        s0 = vfmaq_f32(s0, t0, t0); s1 = vfmaq_f32(s1, t1, t1);
        s2 = vfmaq_f32(s2, t2, t2); s3 = vfmaq_f32(s3, t3, t3);
    }
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; i < D; ++i) { float t = a[i] - b[i]; s += t * t; }
    return s;
#elif defined(__AVX2__) && defined(__FMA__)
    __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
    int i = 0;
    for (; i + 32 <= D; i += 32) {
        __m256 t0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        __m256 t1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
        __m256 t2 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16));
        __m256 t3 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24));
        s0 = _mm256_fmadd_ps(t0, t0, s0); s1 = _mm256_fmadd_ps(t1, t1, s1);
        s2 = _mm256_fmadd_ps(t2, t2, s2); s3 = _mm256_fmadd_ps(t3, t3, s3);
    }
    for (; i + 8 <= D; i += 8) {
        __m256 t0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        s0 = _mm256_fmadd_ps(t0, t0, s0);
    }
    float s = hsum256(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
    for (; i < D; ++i) { float t = a[i] - b[i]; s += t * t; }
    return s;
#else
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int i = 0;
    for (; i + 4 <= D; i += 4) {
        float t0 = a[i] - b[i], t1 = a[i + 1] - b[i + 1];
        float t2 = a[i + 2] - b[i + 2], t3 = a[i + 3] - b[i + 3];
        s0 += t0 * t0; s1 += t1 * t1; s2 += t2 * t2; s3 += t3 * t3;
    }
    for (; i < D; ++i) { float t = a[i] - b[i]; s0 += t * t; }
    return (s0 + s1) + (s2 + s3);
#endif
}

inline float dot(const float* a, const float* b, int D) {
#if defined(__ARM_NEON)
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
    int i = 0;
    for (; i + 16 <= D; i += 16) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
        s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; i < D; ++i) s += a[i] * b[i];
    return s;
#elif defined(__AVX2__) && defined(__FMA__)
    __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
    int i = 0;
    for (; i + 32 <= D; i += 32) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
    }
    for (; i + 8 <= D; i += 8) s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    float s = hsum256(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
    for (; i < D; ++i) s += a[i] * b[i];
    return s;
#else
    float s = 0;
    for (int i = 0; i < D; ++i) s += a[i] * b[i];
    return s;
#endif
}

// sum over dims [0,w) of (qa[i] - code[i]*step[i])^2 ; qa = q_head - lo
inline float sq_dist_codes(const float* qa, const float* step, const int8_t* code, int w) {
#if defined(__ARM_NEON)
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0;
    int i = 0;
    for (; i + 16 <= w; i += 16) {
        int8x16_t c = vld1q_s8(code + i);
        int16x8_t lo = vmovl_s8(vget_low_s8(c)), hi = vmovl_s8(vget_high_s8(c));
        float32x4_t c0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo)));
        float32x4_t c1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo)));
        float32x4_t c2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi)));
        float32x4_t c3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi)));
        float32x4_t t0 = vfmsq_f32(vld1q_f32(qa + i), c0, vld1q_f32(step + i));
        float32x4_t t1 = vfmsq_f32(vld1q_f32(qa + i + 4), c1, vld1q_f32(step + i + 4));
        float32x4_t t2 = vfmsq_f32(vld1q_f32(qa + i + 8), c2, vld1q_f32(step + i + 8));
        float32x4_t t3 = vfmsq_f32(vld1q_f32(qa + i + 12), c3, vld1q_f32(step + i + 12));
        s0 = vfmaq_f32(s0, t0, t0); s1 = vfmaq_f32(s1, t1, t1);
        s0 = vfmaq_f32(s0, t2, t2); s1 = vfmaq_f32(s1, t3, t3);
    }
    float s = vaddvq_f32(vaddq_f32(s0, s1));
    for (; i < w; ++i) { float t = qa[i] - (float)code[i] * step[i]; s += t * t; }
    return s;
#elif defined(__AVX2__) && defined(__FMA__)
    __m256 s0 = _mm256_setzero_ps(), s1 = s0;
    int i = 0;
    for (; i + 16 <= w; i += 16) {
        __m256 t0 = _mm256_fnmadd_ps(load_i8_as_ps(code + i), _mm256_loadu_ps(step + i), _mm256_loadu_ps(qa + i));
        __m256 t1 = _mm256_fnmadd_ps(load_i8_as_ps(code + i + 8), _mm256_loadu_ps(step + i + 8), _mm256_loadu_ps(qa + i + 8));
        s0 = _mm256_fmadd_ps(t0, t0, s0); s1 = _mm256_fmadd_ps(t1, t1, s1);
    }
    for (; i + 8 <= w; i += 8) {
        __m256 t0 = _mm256_fnmadd_ps(load_i8_as_ps(code + i), _mm256_loadu_ps(step + i), _mm256_loadu_ps(qa + i));
        s0 = _mm256_fmadd_ps(t0, t0, s0);
    }
    float s = hsum256(_mm256_add_ps(s0, s1));
    for (; i < w; ++i) { float t = qa[i] - (float)code[i] * step[i]; s += t * t; }
    return s;
#else
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int i = 0;
    for (; i + 4 <= w; i += 4) {
        float t0 = qa[i] - (float)code[i] * step[i];
        float t1 = qa[i + 1] - (float)code[i + 1] * step[i + 1];
        float t2 = qa[i + 2] - (float)code[i + 2] * step[i + 2];
        float t3 = qa[i + 3] - (float)code[i + 3] * step[i + 3];
        s0 += t0 * t0; s1 += t1 * t1; s2 += t2 * t2; s3 += t3 * t3;
    }
    for (; i < w; ++i) { float t = qa[i] - (float)code[i] * step[i]; s0 += t * t; }
    return (s0 + s1) + (s2 + s3);
#endif
}

// Router over int8 codes. At build / insert both arguments are int8 rows; at search
// the first argument is the float query in code units ((q_h - lo)/step). Distances
// are weighted by step^2 per dim so they are in the original head units.
struct RouterParam { size_t dim; const float* step2; bool query_float; };

inline float dist_i8_i8(const int8_t* a, const int8_t* b, const float* w, int d) {
#if defined(__ARM_NEON)
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0;
    int i = 0;
    for (; i + 16 <= d; i += 16) {
        int8x16_t da = vld1q_s8(a + i), db = vld1q_s8(b + i);
        int16x8_t lo = vsubl_s8(vget_low_s8(da), vget_low_s8(db));
        int16x8_t hi = vsubl_s8(vget_high_s8(da), vget_high_s8(db));
        float32x4_t t0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo)));
        float32x4_t t1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo)));
        float32x4_t t2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi)));
        float32x4_t t3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi)));
        s0 = vfmaq_f32(s0, vmulq_f32(t0, t0), vld1q_f32(w + i));
        s1 = vfmaq_f32(s1, vmulq_f32(t1, t1), vld1q_f32(w + i + 4));
        s0 = vfmaq_f32(s0, vmulq_f32(t2, t2), vld1q_f32(w + i + 8));
        s1 = vfmaq_f32(s1, vmulq_f32(t3, t3), vld1q_f32(w + i + 12));
    }
    float s = vaddvq_f32(vaddq_f32(s0, s1));
    for (; i < d; ++i) { float t = (float)a[i] - (float)b[i]; s += w[i] * t * t; }
    return s;
#elif defined(__AVX2__) && defined(__FMA__)
    __m256 s0 = _mm256_setzero_ps(), s1 = s0;
    int i = 0;
    for (; i + 16 <= d; i += 16) {
        __m256 t0 = _mm256_sub_ps(load_i8_as_ps(a + i), load_i8_as_ps(b + i));
        __m256 t1 = _mm256_sub_ps(load_i8_as_ps(a + i + 8), load_i8_as_ps(b + i + 8));
        s0 = _mm256_fmadd_ps(_mm256_mul_ps(t0, t0), _mm256_loadu_ps(w + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_mul_ps(t1, t1), _mm256_loadu_ps(w + i + 8), s1);
    }
    float s = hsum256(_mm256_add_ps(s0, s1));
    for (; i < d; ++i) { float t = (float)a[i] - (float)b[i]; s += w[i] * t * t; }
    return s;
#else
    float s = 0;
    for (int i = 0; i < d; ++i) { float t = (float)a[i] - (float)b[i]; s += w[i] * t * t; }
    return s;
#endif
}

inline float dist_f_i8(const float* a, const int8_t* b, const float* w, int d) {
#if defined(__ARM_NEON)
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0;
    int i = 0;
    for (; i + 16 <= d; i += 16) {
        int8x16_t c = vld1q_s8(b + i);
        int16x8_t lo = vmovl_s8(vget_low_s8(c)), hi = vmovl_s8(vget_high_s8(c));
        float32x4_t t0 = vsubq_f32(vld1q_f32(a + i), vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))));
        float32x4_t t1 = vsubq_f32(vld1q_f32(a + i + 4), vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))));
        float32x4_t t2 = vsubq_f32(vld1q_f32(a + i + 8), vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))));
        float32x4_t t3 = vsubq_f32(vld1q_f32(a + i + 12), vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))));
        s0 = vfmaq_f32(s0, vmulq_f32(t0, t0), vld1q_f32(w + i));
        s1 = vfmaq_f32(s1, vmulq_f32(t1, t1), vld1q_f32(w + i + 4));
        s0 = vfmaq_f32(s0, vmulq_f32(t2, t2), vld1q_f32(w + i + 8));
        s1 = vfmaq_f32(s1, vmulq_f32(t3, t3), vld1q_f32(w + i + 12));
    }
    float s = vaddvq_f32(vaddq_f32(s0, s1));
    for (; i < d; ++i) { float t = a[i] - (float)b[i]; s += w[i] * t * t; }
    return s;
#elif defined(__AVX2__) && defined(__FMA__)
    __m256 s0 = _mm256_setzero_ps(), s1 = s0;
    int i = 0;
    for (; i + 16 <= d; i += 16) {
        __m256 t0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), load_i8_as_ps(b + i));
        __m256 t1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), load_i8_as_ps(b + i + 8));
        s0 = _mm256_fmadd_ps(_mm256_mul_ps(t0, t0), _mm256_loadu_ps(w + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_mul_ps(t1, t1), _mm256_loadu_ps(w + i + 8), s1);
    }
    float s = hsum256(_mm256_add_ps(s0, s1));
    for (; i < d; ++i) { float t = a[i] - (float)b[i]; s += w[i] * t * t; }
    return s;
#else
    float s = 0;
    for (int i = 0; i < d; ++i) { float t = a[i] - (float)b[i]; s += w[i] * t * t; }
    return s;
#endif
}

static float router_dist(const void* a, const void* b, const void* prm) {
    const RouterParam* p = (const RouterParam*)prm;
    return p->query_float ? dist_f_i8((const float*)a, (const int8_t*)b, p->step2, (int)p->dim)
                          : dist_i8_i8((const int8_t*)a, (const int8_t*)b, p->step2, (int)p->dim);
}

struct RouterSpace : public hnswlib::SpaceInterface<float> {
    RouterParam prm;
    std::vector<float> step2;
    RouterSpace(size_t dim, const float* step) : step2(step, step + dim) {
        for (auto& v : step2) v = v * v;
        prm = {dim, step2.data(), false};
    }
    size_t get_data_size() override { return prm.dim; }  // int8 codes
    hnswlib::DISTFUNC<float> get_dist_func() override { return router_dist; }
    void* get_dist_func_param() override { return &prm; }
};

// Purpose-built router search: hnswlib builds the graph (and handles inserts); the
// level-0 links and int8 codes are extracted into a flat, pos-major layout and searched
// with a beam search that uses an int8 x int8 dot product (SDOT) against a quantised
// query, prefetching neighbour codes. Only the *ordering* of hits matters for routing,
// so the query quantisation error is acceptable (the cascade bounds stay exact).
// The router graph the beam search reads is always a FlatGraph (csrc/flat_graph.h): an owned index
// (hnswlib built it) extracts one into `buf` after a build or inserts (refresh_graph); a frozen index
// reads the flat image's graph section in place, plus an optional delta graph the extension builds
// over the positions appended since the image (cascade_set_delta_graph). Node ids are positions
// (delta graph: position - nb). hnswlib numbers nodes in arrival order (internal id); with serial
// insertion that is the label (pos), with a threaded build it is not — refresh_graph maps.
struct RouterGraph {
    std::vector<char> buf;        // owned mode: the extracted FlatGraph
    std::vector<int32_t> int2pos, pos2int;
    bool dirty = true;
    FlatGraph* owned() { return buf.empty() ? nullptr : (FlatGraph*)buf.data(); }
};

inline int32_t dot_i8(const int8_t* a, const int8_t* b, int d) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    int32x4_t s0 = vdupq_n_s32(0), s1 = s0;
    int i = 0;
    for (; i + 32 <= d; i += 32) {
        s0 = vdotq_s32(s0, vld1q_s8(a + i), vld1q_s8(b + i));
        s1 = vdotq_s32(s1, vld1q_s8(a + i + 16), vld1q_s8(b + i + 16));
    }
    for (; i + 16 <= d; i += 16) s0 = vdotq_s32(s0, vld1q_s8(a + i), vld1q_s8(b + i));
    int32_t s = vaddvq_s32(vaddq_s32(s0, s1));
    for (; i < d; ++i) s += (int32_t)a[i] * (int32_t)b[i];
    return s;
#elif defined(__AVX2__) && defined(__FMA__)
    __m256i acc = _mm256_setzero_si256();
    int i = 0;
    for (; i + 16 <= d; i += 16) {
        __m256i va = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(a + i)));
        __m256i vb = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(b + i)));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(va, vb));
    }
    __m128i lo = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    lo = _mm_hadd_epi32(lo, lo); lo = _mm_hadd_epi32(lo, lo);
    int32_t s = _mm_cvtsi128_si32(lo);
    for (; i < d; ++i) s += (int32_t)a[i] * (int32_t)b[i];
    return s;
#else
    int32_t s = 0;
    for (int i = 0; i < d; ++i) s += (int32_t)a[i] * (int32_t)b[i];
    return s;
#endif
}

inline float lower_bound2(float head_d2, float err, float a, float b, float c) {
    float hl = std::sqrt(std::max(head_d2, 0.f)) - err;
    if (hl < 0) hl = 0;
    float t = a * a + b * b - 2.f * c * a * b;
    if (t < 0) t = 0;
    return hl * hl + t;
}

// Flat (frozen) image: the PostgreSQL extension's shared segment (E1). Position-independent — a header
// of byte offsets from its own start, every array 64-byte aligned — so several processes map one copy
// and search it in place. cascade_flatten writes it from an owned index (cells fully repacked first:
// no overflow blocks, no tombstoned rows in the blocks); cascade_open_flat returns an Index whose base
// reads it and whose deltas (inserts, tombstones since the image) are private to the process. The
// router's upper layers are stored flat too (pos space), so the frozen index needs no hnswlib object
// for the base; inserted points go to a small per-process hnswlib graph searched beside the flat beam.
// FlatHeader: csrc/flat_image.h (shared with the extension).
template <class T> inline const T* flat_ptr(const FlatHeader* h, uint64_t off) { return (const T*)((const char*)h + off); }

struct Index {
    int n0 = 0, D = 0, C = 0, L = 0;     // n0 = points in the main blocks
    // Frozen base (cascade_open_flat): positions [0, nb) read the flat image in place and the vectors
    // below hold only positions >= nb (this process's delta); pos_of splits at fb_npos_of the same way.
    // An owned index has flat == nullptr and nb == 0, so every accessor reads the vectors.
    const FlatHeader* flat = nullptr;
    int nb = 0, fb_npos_of = 0;
    const int32_t *fb_perm = nullptr, *fb_pos_of = nullptr, *fb_cell_of = nullptr, *fb_blk_pos = nullptr;
    const uint64_t *fb_bco = nullptr, *fb_bpo = nullptr;
    const int8_t* fb_bcodes = nullptr;
    const FlatGraph* bgraph = nullptr;   // frozen: the image's graph section; owned: rg.owned() after refresh_graph
    const FlatGraph* dgraph = nullptr;   // frozen: the extension's private graph over positions nb.. (ids + nb), may be null
    int dgraph_nb = 0;                   // the position the delta graph's id 0 stands for
    int n_delta_live = 0;                // frozen: appended positions not tombstoned (the delta graph is searched only while > 0)
    const float *fb_err[4] = {nullptr, nullptr, nullptr, nullptr}, *fb_tail[4] = {nullptr, nullptr, nullptr, nullptr};
    int perm_at(int pos) const { return pos < nb ? fb_perm[pos] : perm[pos - nb]; }
    int cell_at(int pos) const { return pos < nb ? fb_cell_of[pos] : cell_of[pos - nb]; }
    float err_at(int l, int pos) const { return pos < nb ? fb_err[l][pos] : err[l][pos - nb]; }
    float tail_at(int l, int pos) const { return pos < nb ? fb_tail[l][pos] : tail[l][pos - nb]; }
    int pos_of_id(int id) const {
        if (id < 0) return -1;
        if (id < fb_npos_of) return fb_pos_of[id];
        id -= fb_npos_of;
        return id < (int)pos_of.size() ? pos_of[id] : -1;
    }
    int blk_cnt(int c) const { return flat ? (int)(fb_bpo[c + 1] - fb_bpo[c]) : (int)blk_pos[c].size(); }
    const int8_t* blk_codes_ptr(int c) const { return flat ? fb_bcodes + fb_bco[c] : blk_codes[c].data(); }
    const int* blk_pos_ptr(int c) const { return flat ? fb_blk_pos + fb_bpo[c] : blk_pos[c].data(); }
    int lv[4] = {0, 0, 0, 0};            // cumulative level dims, lv[L-1] = d3
    std::vector<std::vector<int8_t>> blk_codes;  // per cell: level-blocked codes of its main block
    std::vector<std::vector<int>> blk_pos;       // per cell: positions of the main block's points
    std::vector<int> perm;               // pos -> original id (grows with inserts)
    std::vector<int> pos_of;             // original id -> pos
    std::vector<int> cell_of;            // pos -> cell
    std::vector<char> deleted;           // pos -> tombstone
    std::vector<std::vector<int8_t>> ov_codes;  // per cell: row-major d3 codes of inserted points
    std::vector<std::vector<int>> ov_pos;       // per cell: their positions
    std::vector<float> lo, step;
    std::vector<float> err[4], tail[4];  // per level, indexed by pos
    std::vector<float> xfull;            // pos-major copy, original space (copy mode)
    const float* xext = nullptr;         // borrow mode: caller's id-major array for ids < n0
    std::vector<float> xins;             // borrow mode: inserted vectors, (pos - n0)-major
    const float* vec(int pos) const {
        if (!xext) return &xfull[(size_t)pos * D];
        return pos < n0 ? xext + (size_t)perm[pos] * D : &xins[(size_t)(pos - n0) * D];
    }
    std::vector<float> proj, mean;
    std::vector<float> cent;             // C*d3 head centroids (fallback ordering, insert assignment)
    float ch[4] = {1, 1, 1, 1};
    RouterSpace* space = nullptr;
    hnswlib::HierarchicalNSW<float>* hnsw = nullptr;
    long long n_inserted = 0, n_deleted = 0, n_compacted = 0;
    // flat-router cell selection: 0 = cells of the top-M hits (default); 1 = cells of all ef results
    // (the beam's result heap, ef >= M); 2 = cells of every point the beam scored (visited set)
    int router_mode = 0;
    int profile = 0;  // 1: accumulate per-phase nanoseconds into stats[3..8] (see search_one)
    // reduced router (experiment T6): the router graph is built over n_rnodes node sketches
    // (sub-centroids or a subsample) instead of every point; rnode_cell maps a node to its cell.
    // Inserts/deletes do not update a reduced router.
    int n_rnodes = 0;
    std::vector<int> rnode_cell;
    bool reduced() const { return n_rnodes > 0; }
    int router_cell(int node) const { return n_rnodes ? rnode_cell[node] : cell_at(node); }
    RouterGraph rg;
    std::vector<int8_t> rcodes;          // pos-major int8 heads (router copy)
    // R1a (router levers, 2026-10-27): scoring width. 0 = today's behaviour (every visited node is scored on
    // the full d3-wide sketch); w < d3 = the beam scores the first w coordinates (the leading PCA axes carry
    // most of the energy) and re-scores its ef results at full width before the cells are ranked. Base graph
    // only (a delta graph is always scored at full width); sc_w caches the partial norms per node.
    int router_width = 0;
    std::vector<float> sc_w;
    const FlatGraph* sc_w_for = nullptr;
    int sc_w_width = 0;
    // R1c: seeded beam. A second router graph over a few node sketches (sub-centroids, cascade_set_seed_router)
    // ranks cells; the medoids of its top seed_C cells are the per-point beam's entry points instead of the
    // single global entry point (seed_C = 0: today's behaviour). Owned, per-point indexes only.
    hnswlib::HierarchicalNSW<float>* seed_hnsw = nullptr;
    std::vector<char> seed_buf;          // FlatGraph over the seed nodes
    std::vector<int8_t> seed_codes;
    std::vector<int> seed_cell;          // seed node -> cell
    std::vector<int> cell_medoid;        // cell -> position of its point nearest the head centroid (-1: none)
    int seed_C = 0, seed_ef = 0;
    const FlatGraph* seed_graph() const { return seed_buf.empty() ? nullptr : (const FlatGraph*)seed_buf.data(); }
    // conformal calibration (cascade_build_vectors): per level the sorted critical cosines g* of the
    // calibration pairs; c_h(delta) is their (1 - delta) quantile on the [-1, 1] grid (ch_for_delta)
    std::vector<float> cal[4];
    float var_captured = 0;              // share of the sample variance in the d3-dim head (fit_head)
    bool has_x() const { return xext != nullptr || !xfull.empty(); }
    int total() const { return nb + (int)perm.size(); }
    ~Index() { delete hnsw; delete seed_hnsw; delete space; }   // the flat image belongs to the caller
};

// Software prefetch distances for the cascade scan (survivor codes) and the exact phase (full rows).
// At 1M x 1536 the structure no longer fits in cache and both loops are DRAM-latency-bound without it.
#ifndef PF_AHEAD
#define PF_AHEAD 8
#endif
#ifndef PF_ROWS
#define PF_ROWS 2
#endif

inline long long now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Scratch {
    std::vector<float> qc, qh, qa, key, head_d2;
    std::vector<int> alive, alive2;
    std::vector<std::pair<float, int>> cells, cand;
    std::vector<float> cell_score;
    std::vector<char> cell_hit;
    std::vector<int8_t> q8;
    std::vector<std::pair<float, int>> beam, res;
    // visited stamps live in the per-call scratch (not in the shared graph) so several
    // threads can search the same index concurrently (each cascade_search call has its own)
    std::vector<int32_t> visited;
    int32_t epoch = 0;
    std::vector<std::pair<float, int>> scored;  // router_mode 2: every (score, pos) the beam evaluated
    float a[4] = {0, 0, 0, 0};           // per-level tail norm of the current query (route_query)
    long long ops = 0;                   // dimension-ops of the current query
    // P5a: an image value a page corruption could have put out of range (a hit's cell, a blocked position): the
    // scan entry points throw "corrupt image: ..." (the extension reports ERRCODE_INDEX_CORRUPTED) instead of
    // indexing with it; the deep checker finds the same values
    char corrupt[96] = {0};
};

// Extract hnswlib's graph into a FlatGraph blob in pos space (after a build or inserts; owned indexes).
static void refresh_graph(Index* ix) {
    RouterGraph& g = ix->rg;
    const int d3 = ix->lv[ix->L - 1];
    const int n = ix->reduced() ? ix->n_rnodes : ix->total();
    auto* h = ix->hnsw;
    const int M = (int)h->M_;
    g.int2pos.assign(n, -1); g.pos2int.assign(n, -1);
    for (int i = 0; i < n; ++i) {
        const int pos = (int)h->getExternalLabel((hnswlib::tableint)i);
        g.int2pos[i] = pos; g.pos2int[pos] = i;
    }
    // upper blocks needed: sum of levels
    int64_t blocks = 0;
    for (int i = 0; i < n; ++i) blocks += std::max(0, h->element_levels_[(size_t)i]);
    FlatGraph hd;
    const int32_t up_cap = (int32_t)std::max<int64_t>(blocks, 1);
    const uint64_t bytes = fg_layout(&hd, std::max(n, 1), d3, M, up_cap);   // an empty index keeps a one-node area (H14)
    g.buf.assign(bytes, 0);
    FlatGraph* fg = (FlatGraph*)g.buf.data();
    *fg = hd;
    fg->n = n; fg->efC = (int)h->ef_construction_; fg->maxlevel = h->maxlevel_;
    fg->enterpoint = n ? g.int2pos[(int)h->enterpoint_node_] : -1;
    fg->seed = 0; fg->mult = 1.0 / std::log(std::max(2, M));
    const float* w = ix->space->step2.data();
    std::memcpy(fg_w2(fg), w, sizeof(float) * d3);
    if (n > 0) std::memcpy(fg_codes(fg), ix->rcodes.data(), (size_t)n * d3);   // an empty vector's data() is null: UB for memcpy even at size 0
    float* sc = fg_sc(fg);
    for (int pos = 0; pos < n; ++pos) {
        const int8_t* c = fg_code(fg, pos);
        float acc = 0;
        for (int i = 0; i < d3; ++i) acc += w[i] * (float)c[i] * (float)c[i];
        sc[pos] = acc;
    }
    int32_t* lvl = fg_level(fg); int32_t* upidx = fg_upidx(fg);
    int32_t next_block = 0;
    for (int pos = 0; pos < n; ++pos) {
        const hnswlib::tableint ii = (hnswlib::tableint)g.pos2int[pos];
        const int L = h->element_levels_[(size_t)ii];
        lvl[pos] = L;
        auto* ll0 = h->get_linklist0(ii);
        int cnt = (int)h->getListCount(ll0);
        auto* data = (hnswlib::tableint*)(ll0 + 1);
        int32_t* dst = fg_links0(fg, pos);
        int k = 0;
        for (int j = 0; j < cnt && k < fg->M0; ++j) dst[1 + k++] = g.int2pos[(int)data[j]];
        dst[0] = k;
        upidx[pos] = L > 0 ? next_block : -1;
        for (int level = 1; level <= L; ++level) {
            auto* ll = h->get_linklist(ii, level);
            int c2 = (int)h->getListCount(ll);
            auto* d2 = (hnswlib::tableint*)(ll + 1);
            int32_t* blk = (int32_t*)(g.buf.data() + fg->uplinks_off) + (size_t)next_block * (fg->M + 1);
            int kk = 0;
            for (int j = 0; j < c2 && kk < fg->M; ++j) blk[1 + kk++] = g.int2pos[(int)d2[j]];
            blk[0] = kk;
            ++next_block;
        }
    }
    fg->up_used = next_block;
    ix->bgraph = fg;
    g.dirty = false;
}

// Beam search on a flat graph. Query: v_i = w_i * key_i quantised to int8 with scale qs (one scale per
// level block: PCA variance decays by level), score(c) = sc[c] - 2 qs <v8, c>  (= ||key - c||_w^2 up to a
// per-query constant, the same for every graph searched with the same key). Results land in s.res as
// (score, position) with position = node id + id_base (the delta graph's ids start at nb).
// R1a: the partial norms sum_{i < w} w_i c_i^2 of every node of g (the beam's per-node constant at width w).
static void ensure_sc_w(Index* ix, const FlatGraph* g, int w) {
    if (ix->sc_w_for == g && ix->sc_w_width == w && (int)ix->sc_w.size() == g->n_cap) return;
    const float* w2 = fg_w2_c(g);
    ix->sc_w.assign(g->n_cap, 0.f);
    for (int id = 0; id < g->n_cap; ++id) {
        const int8_t* c = fg_code(g, id);
        float acc = 0;
        for (int i = 0; i < w; ++i) acc += w2[i] * (float)c[i] * (float)c[i];
        ix->sc_w[id] = acc;
    }
    ix->sc_w_for = g; ix->sc_w_width = w;
}

// entries / n_entries (R1c): start the level-0 beam from these node ids instead of the greedy descent from the
// graph's entry point. deleted_check: false for a graph whose ids are not positions (the seed router).
// dead_bound: the beam's termination bound is the ef-th best score over every node it scored, live or
// tombstoned (res still holds live nodes only). Without it a graph whose nodes are mostly tombstoned
// never fills res, the bound never tightens and the beam walks the whole graph (the extension's delta
// graph after a VACUUM). The default keeps the previous behaviour bit for bit.
static void graph_search(Index* ix, const FlatGraph* g, Scratch& s, const float* key, int ef, int M, int id_base, bool append,
                         const int* entries = nullptr, int n_entries = 0, bool deleted_check = true, bool dead_bound = false) {
    const int d3 = g->d3, deg = g->M0;
    const float* w = fg_w2_c(g);
    s.q8.resize(d3);
    const int L = ix->L;
    float qs[4] = {1, 1, 1, 1};
    int lo_d[4], wd[4], wdW[4];
    // scoring width: the base graph at router_width (R1a), everything else at full width
    const int W = (ix->router_width > 0 && ix->router_width < d3 && g == ix->bgraph) ? ix->router_width : d3;
    if (W < d3) ensure_sc_w(ix, g, W);
    for (int l = 0; l < L; ++l) {
        lo_d[l] = l ? ix->lv[l - 1] : 0; wd[l] = ix->lv[l] - lo_d[l];
        wdW[l] = std::max(0, std::min(ix->lv[l], W) - lo_d[l]);
        float vmax = 0;
        for (int i = lo_d[l]; i < ix->lv[l]; ++i) vmax = std::max(vmax, std::fabs(w[i] * key[i]));
        qs[l] = vmax > 0 ? vmax / 127.f : 1.f;
        for (int i = lo_d[l]; i < ix->lv[l]; ++i) s.q8[i] = (int8_t)std::lrint(w[i] * key[i] / qs[l]);
    }
    const bool record = ix->router_mode == 2;
    if (!append) { s.scored.clear(); s.res.clear(); }
    // H17: a delta graph whose nodes are not linked yet (enterpoint < 0; the extension links only once the
    // delta outgrows trigpoint.delta_flat_max) is scored flat: every present node, the ef best live ones
    const bool flat = g == ix->dgraph && g->enterpoint < 0;
    if (g->n <= 0 || (g->enterpoint < 0 && !flat)) return;
    const float* sc = fg_sc_c(g);
    const float* scW = W < d3 ? ix->sc_w.data() : sc;
    const int32_t* lvl = fg_level_c(g);
    auto score_full = [&](int id) {
        const int8_t* c = fg_code(g, id);
        float acc = sc[id];
        for (int l = 0; l < L; ++l) acc -= 2.f * qs[l] * (float)dot_i8(s.q8.data() + lo_d[l], c + lo_d[l], wd[l]);
        return acc;
    };
    auto score = [&](int id) {
        const int8_t* c = fg_code(g, id);
        float acc = scW[id];
        for (int l = 0; l < L && wdW[l] > 0; ++l) acc -= 2.f * qs[l] * (float)dot_i8(s.q8.data() + lo_d[l], c + lo_d[l], wdW[l]);
        if (record) s.scored.push_back({acc, id + id_base});
        return acc;
    };
    // visited stamps: sized for the largest graph seen (one epoch space; ids of different graphs never mix in one call)
    if ((int)s.visited.size() < g->n_cap) { s.visited.assign(g->n_cap, 0); s.epoch = 0; }
    if (++s.epoch == 0) { std::fill(s.visited.begin(), s.visited.end(), 0); s.epoch = 1; }
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<>> cand;
    std::priority_queue<std::pair<float, int>> res;   // live results (max-first, at most ef)
    std::priority_queue<std::pair<float, int>> all;   // dead_bound: every scored node, for the termination bound
    auto push_res = [&](float d, int v, bool live) {
        if (live) { res.push({d, v}); if ((int)res.size() > ef) res.pop(); }
        if (dead_bound) { all.push({d, v}); if ((int)all.size() > ef) all.pop(); }
    };
    auto full = [&]() { return dead_bound ? (int)all.size() >= ef : (int)res.size() >= ef; };
    auto top_bound = [&]() { return dead_bound ? all.top().first : res.top().first; };
    if (flat) {
        for (int id = 0; id < g->n; ++id) {
            if (lvl[id] < 0) continue;
            const bool live = !deleted_check || ix->reduced() || !ix->deleted[id + id_base];
            if (!live) continue;
            push_res(score(id), id, true);
        }
    } else {
    float bound;
    if (n_entries > 0) {
        // seeded: every entry is scored and enters both heaps (no descent through the upper layers)
        for (int t = 0; t < n_entries; ++t) {
            const int e = entries[t];
            if (e < 0 || e >= g->n_cap || lvl[e] < 0 || s.visited[e] == s.epoch) continue;
            s.visited[e] = s.epoch;
            const float de = score(e);
            cand.push({de, e});
            push_res(de, e, !deleted_check || ix->reduced() || !ix->deleted[e + id_base]);
        }
        if (cand.empty()) return;
        bound = full() ? top_bound() : INFINITY;
    } else {
        // entry: greedy descent through the upper layers with the same score
        int cur = g->enterpoint;
        float curd = score(cur);
        for (int level = g->maxlevel; level > 0; --level) {
            bool changed = true;
            while (changed) {
                changed = false;
                const int32_t* lk = fg_links_c(g, cur, level);
                if (!lk) break;
                const int cnt = std::min<int>(lk[0], g->M);
                for (int j = 1; j <= cnt; ++j) {
                    const int v = lk[j];
                    if (v < 0 || v >= g->n_cap || lvl[v] < 0) continue;
                    float d = score(v);
                    if (d < curd) { curd = d; cur = v; changed = true; }
                }
            }
        }
        // level 0 beam search: candidates (min-first) and results (max-first), both as heaps
        s.visited[cur] = s.epoch;
        cand.push({curd, cur}); push_res(curd, cur, true);
        bound = curd;
    }
    while (!cand.empty()) {
        auto [d, c] = cand.top();
        if (d > bound && full()) break;
        cand.pop();
        const int32_t* nb = fg_links0_c(g, c);
        const int cnt = std::min<int>(nb[0], deg);
        for (int j = 1; j <= cnt; ++j) if (nb[j] >= 0 && nb[j] < g->n_cap) __builtin_prefetch(fg_code(g, nb[j]));
        for (int j = 1; j <= cnt; ++j) {
            int v = nb[j];
            if (v < 0 || v >= g->n_cap || lvl[v] < 0 || s.visited[v] == s.epoch) continue;
            s.visited[v] = s.epoch;
            float dv = score(v);
            if (!full() || dv < bound) {
                cand.push({dv, v});
                push_res(dv, v, !deleted_check || ix->reduced() || !ix->deleted[v + id_base]);
                if (full()) bound = top_bound();
            }
        }
    }
    }
    std::vector<std::pair<float, int>> out;
    while (!res.empty()) { auto pr = res.top(); res.pop(); out.push_back({pr.first, pr.second + id_base}); }
    std::reverse(out.begin(), out.end());
    if (W < d3) {  // R1a: the ef results re-scored on the whole sketch, then ranked
        for (auto& pr : out) pr.first = score_full(pr.second - id_base);
        std::sort(out.begin(), out.end());
    }
    if (ix->router_mode == 0 && (int)out.size() > M) out.resize(M);
    s.res.insert(s.res.end(), out.begin(), out.end());
}

// quantise one head into signed int8 codes and its per-level err / tail norms
static void encode_point(const Index* ix, const float* x, const float* h, int8_t* row,
                         float* err_out, float* tail_out) {
    const int D = ix->D, L = ix->L;
    float xn2 = 0;
    for (int i = 0; i < D; ++i) { float t = x[i] - ix->mean[i]; xn2 += t * t; }
    float hn2 = 0, e2 = 0;
    int i = 0;
    for (int l = 0; l < L; ++l) {
        for (; i < ix->lv[l]; ++i) {
            float v = (h[i] - ix->lo[i]) / ix->step[i];
            int q = (int)std::lrint(v);
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            row[i] = (int8_t)q;
            float d = h[i] - (ix->lo[i] + (float)q * ix->step[i]);
            e2 += d * d;
            hn2 += h[i] * h[i];
        }
        err_out[l] = std::sqrt(e2);
        tail_out[l] = std::sqrt(std::max(xn2 - hn2, 0.f));
    }
}

}  // namespace

// Cooperative cancellation for long builds (the PostgreSQL extension polls its interrupt flag here;
// nothing in this file may call back into the server). Checked every 1024 points.
typedef int (*abort_fn_t)(void*);
static thread_local abort_fn_t g_abort_fn = nullptr;
static thread_local void* g_abort_ctx = nullptr;
static inline bool build_aborted(int i) { return g_abort_fn && (i & 1023) == 0 && g_abort_fn(g_abort_ctx); }

// Build-time parallelism (the extension's `threads` reloption, default 1): worker threads exist only
// inside cascade_build_vectors' phases, touch nothing but this library's own memory, inherit a
// fully blocked signal mask (so the host process's signal handlers only ever run on the leader),
// never call the abort hook (g_abort_fn is thread_local and null in workers; the leader polls it
// between phases) and are joined before the phase returns. An exception in a worker is captured
// and rethrown by the leader after the join.
static thread_local int g_build_threads = 1;
template <class F>
static void parallel_for(int n, F f) {  // f(thread_id, lo, hi) over [0, n)
    const int T = std::max(1, g_build_threads);
    if (T == 1 || n < 2 * T) { f(0, 0, n); return; }
    std::vector<std::thread> ts;
    std::mutex em; std::string err; std::atomic<bool> failed{false};
    sigset_t all, old;
    sigfillset(&all);
    pthread_sigmask(SIG_BLOCK, &all, &old);
    for (int t = 0; t < T; ++t) {
        const int lo = (int)((long long)n * t / T), hi = (int)((long long)n * (t + 1) / T);
        ts.emplace_back([&, t, lo, hi] {
            try { f(t, lo, hi); }
            catch (const std::exception& e) { std::lock_guard<std::mutex> g(em); err = e.what(); failed = true; }
            catch (...) { std::lock_guard<std::mutex> g(em); err = "unknown exception in build worker"; failed = true; }
        });
    }
    pthread_sigmask(SIG_SETMASK, &old, nullptr);
    for (auto& th : ts) th.join();
    if (failed) throw std::runtime_error(err);
}

extern "C" {

void* cascade_build(int n, int D, int C, int L, const int* levels, const int* labels,
                    const float* x, const float* xr_head, const float* proj, const float* mean,
                    const float* lo, const float* step, const float* ch, int M, int efC, int capacity,
                    int borrow_x) {
    Index* ix = new Index();
    ix->n0 = n; ix->D = D; ix->C = C; ix->L = L;
    for (int l = 0; l < L; ++l) { ix->lv[l] = levels[l]; ix->ch[l] = ch[l]; }
    const int d3 = levels[L - 1];

    std::vector<int> cell_cnt(C, 0), cell_off(C + 1, 0);
    for (int i = 0; i < n; ++i) cell_cnt[labels[i]]++;
    for (int c = 0; c < C; ++c) cell_off[c + 1] = cell_off[c] + cell_cnt[c];
    ix->perm.assign(n, 0);
    ix->pos_of.assign(n, -1);
    ix->cell_of.assign(n, 0);
    ix->deleted.assign(n, 0);
    ix->blk_codes.assign(C, {});
    ix->blk_pos.assign(C, {});
    {
        std::vector<int> fill(cell_off.begin(), cell_off.end() - 1);
        for (int i = 0; i < n; ++i) {
            int pos = fill[labels[i]]++;
            ix->perm[pos] = i;
            ix->pos_of[i] = pos;
            ix->cell_of[pos] = labels[i];
        }
    }
    ix->lo.assign(lo, lo + d3);
    ix->step.assign(step, step + d3);
    ix->proj.assign(proj, proj + (size_t)d3 * D);
    ix->mean.assign(mean, mean + D);
    ix->ov_codes.assign(C, {});
    ix->ov_pos.assign(C, {});
    for (int l = 0; l < L; ++l) { ix->err[l].assign(n, 0.f); ix->tail[l].assign(n, 0.f); }
    if (borrow_x) ix->xext = x; else ix->xfull.assign((size_t)n * D, 0.f);
    ix->cent.assign((size_t)C * d3, 0.f);
    ix->space = new RouterSpace((size_t)d3, step);
    const bool build_router = efC > 0;  // efC <= 0: no per-point router (cascade_set_router_nodes follows)
    if (build_router) ix->hnsw = new hnswlib::HierarchicalNSW<float>(ix->space, (size_t)(n + std::max(capacity, 0)), M, efC);
    ix->rcodes.assign((size_t)n * d3, 0);

    std::atomic<bool> aborted{false};
    for (int c = 0; c < C; ++c) { ix->blk_codes[c].assign((size_t)cell_cnt[c] * d3, 0); ix->blk_pos[c].resize(cell_cnt[c]); }
    parallel_for(C, [&](int, int c_lo, int c_hi) {
        std::vector<int8_t> row(d3);
        float e[4], t[4];
        for (int c = c_lo; c < c_hi; ++c) {
            int off = cell_off[c], cnt = cell_cnt[c];
            int8_t* base = ix->blk_codes[c].data();
            for (int j = 0; j < cnt; ++j) {
                int pos = off + j, id = ix->perm[pos];
                if (build_aborted(pos)) { aborted = true; return; }  // leader only (workers have no hook)
                ix->blk_pos[c][j] = pos;
                const float* h = xr_head + (size_t)id * d3;
                const float* xo = x + (size_t)id * D;
                encode_point(ix, xo, h, row.data(), e, t);
                for (int l = 0; l < L; ++l) {
                    int lo_d = l ? ix->lv[l - 1] : 0, w = ix->lv[l] - lo_d;
                    std::memcpy(base + (size_t)cnt * lo_d + (size_t)j * w, row.data() + lo_d, w);
                    ix->err[l][pos] = e[l];
                    ix->tail[l][pos] = t[l];
                }
                if (!borrow_x) std::memcpy(&ix->xfull[(size_t)pos * D], xo, sizeof(float) * D);
                std::memcpy(&ix->rcodes[(size_t)pos * d3], row.data(), d3);
                for (int k = 0; k < d3; ++k) ix->cent[(size_t)c * d3 + k] += h[k];
                if (build_router) ix->hnsw->addPoint(row.data(), (size_t)pos);
            }
            if (cnt) for (int k = 0; k < d3; ++k) ix->cent[(size_t)c * d3 + k] /= (float)cnt;
        }
    });
    if (aborted) { delete ix; return nullptr; }
    ix->space->prm.query_float = true;
    return ix;
}

void cascade_free(void* p) { delete (Index*)p; }

// Insert points (original space) with new integer ids.
void cascade_insert(void* p, const float* xs, const int* ids, int count) {
    Index* ix = (Index*)p;
    const int D = ix->D, L = ix->L, d3 = ix->lv[L - 1], C = ix->C;
    std::vector<float> qc(D), h(d3);
    std::vector<int8_t> row(d3);
    float e[4], t[4];
    ix->space->prm.query_float = false;
    for (int i = 0; i < count; ++i) {
        const float* x = xs + (size_t)i * D;
        for (int k = 0; k < D; ++k) qc[k] = x[k] - ix->mean[k];
        for (int r = 0; r < d3; ++r) h[r] = dot(&ix->proj[(size_t)r * D], qc.data(), D);
        int best = 0; float bd = INFINITY;
        for (int c = 0; c < C; ++c) {
            float d = sq_dist_full(h.data(), &ix->cent[(size_t)c * d3], d3);
            if (d < bd) { bd = d; best = c; }
        }
        encode_point(ix, x, h.data(), row.data(), e, t);
        int pos = ix->total();
        int id = ids[i];
        if (id >= (int)ix->pos_of.size()) ix->pos_of.resize((size_t)id + 1, -1);
        ix->pos_of[id] = pos;
        ix->perm.push_back(id);
        ix->cell_of.push_back(best);
        ix->deleted.push_back(0);
        for (int l = 0; l < L; ++l) { ix->err[l].push_back(e[l]); ix->tail[l].push_back(t[l]); }
        if (ix->xext) ix->xins.insert(ix->xins.end(), x, x + D); else ix->xfull.insert(ix->xfull.end(), x, x + D);
        ix->ov_codes[best].insert(ix->ov_codes[best].end(), row.begin(), row.end());
        ix->ov_pos[best].push_back(pos);
        if ((size_t)pos >= ix->hnsw->max_elements_) ix->hnsw->resizeIndex(ix->hnsw->max_elements_ * 2);
        if (!ix->reduced() && ix->hnsw) ix->hnsw->addPoint(row.data(), (size_t)pos);
        if (!ix->reduced()) ix->rcodes.insert(ix->rcodes.end(), row.begin(), row.end());
        ix->rg.dirty = true;
        ix->n_inserted++;
    }
    ix->space->prm.query_float = true;
}

void cascade_delete(void* p, const int* ids, int count) {
    Index* ix = (Index*)p;
    for (int i = 0; i < count; ++i) {
        int id = ids[i];
        int pos = ix->pos_of_id(id);
        if (pos < 0) continue;
        if (ix->deleted[pos]) continue;
        ix->deleted[pos] = 1;
        ix->n_deleted++;
        if (ix->flat && pos >= ix->nb) ix->n_delta_live--;
        if (!ix->flat && !ix->reduced() && ix->hnsw) { try { ix->hnsw->markDelete((size_t)pos); } catch (...) {} }
        // the flat graph checks ix->deleted directly
    }
}

// Repack one cell: live points of the main block + overflow -> a fresh level-blocked
// main block; tombstoned points disappear from the scan. Positions are unchanged.
static void compact_cell(Index* ix, int c) {
    const int L = ix->L, d3 = ix->lv[L - 1];
    std::vector<int> keep;
    std::vector<int8_t> rows;  // row-major d3 per kept point
    const auto& bp = ix->blk_pos[c];
    const int cnt = (int)bp.size();
    const int8_t* base = ix->blk_codes[c].data();
    std::vector<int8_t> row(d3);
    for (int j = 0; j < cnt; ++j) {
        if (ix->deleted[bp[j]]) continue;
        for (int l = 0; l < L; ++l) {
            int lo_d = l ? ix->lv[l - 1] : 0, w = ix->lv[l] - lo_d;
            std::memcpy(row.data() + lo_d, base + (size_t)cnt * lo_d + (size_t)j * w, w);
        }
        keep.push_back(bp[j]);
        rows.insert(rows.end(), row.begin(), row.end());
    }
    const auto& op = ix->ov_pos[c];
    for (size_t j = 0; j < op.size(); ++j) {
        if (ix->deleted[op[j]]) continue;
        keep.push_back(op[j]);
        rows.insert(rows.end(), ix->ov_codes[c].begin() + (size_t)j * d3, ix->ov_codes[c].begin() + (size_t)(j + 1) * d3);
    }
    const int m = (int)keep.size();
    std::vector<int8_t> blk((size_t)m * d3);
    for (int j = 0; j < m; ++j)
        for (int l = 0; l < L; ++l) {
            int lo_d = l ? ix->lv[l - 1] : 0, w = ix->lv[l] - lo_d;
            std::memcpy(blk.data() + (size_t)m * lo_d + (size_t)j * w, rows.data() + (size_t)j * d3 + lo_d, w);
        }
    ix->blk_codes[c].swap(blk);
    ix->blk_pos[c].swap(keep);
    ix->ov_codes[c].clear(); ix->ov_codes[c].shrink_to_fit();
    ix->ov_pos[c].clear(); ix->ov_pos[c].shrink_to_fit();
    ix->n_compacted++;
}

// Compact every cell whose overflow or tombstones exceed `frac` of its main block.
int cascade_compact(void* p, float frac) {
    Index* ix = (Index*)p;
    int done = 0;
    if (ix->flat) return -1;  // a frozen index is read in place; compaction works on an owned copy
    for (int c = 0; c < ix->C; ++c) {
        const auto& bp = ix->blk_pos[c];
        int dead = 0;
        for (int pos : bp) dead += ix->deleted[pos];
        for (int pos : ix->ov_pos[c]) dead += ix->deleted[pos];
        double load = (double)(ix->ov_pos[c].size() + dead) / std::max<double>(1.0, (double)bp.size());
        if (load > frac && (ix->ov_pos[c].size() + dead) > 0) { compact_cell(ix, c); ++done; }
    }
    return done;
}

int cascade_is_deleted(void* p, int id) {  // 1 if the id is tombstoned or unknown
    Index* ix = (Index*)p;
    int pos = ix->pos_of_id(id);
    return (pos < 0 || ix->deleted[pos]) ? 1 : 0;
}

long long cascade_counts(void* p, long long* out) {  // total, inserted, deleted
    Index* ix = (Index*)p;
    out[0] = ix->total(); out[1] = ix->n_inserted; out[2] = ix->n_deleted;
    return out[0];
}

struct Prof {  // per-phase timing (ix->profile): slots as documented in search_one
    bool on = false; long long t = 0; long long acc[6] = {0, 0, 0, 0, 0, 0};
    void start() { if (on) t = now_ns(); }
    void lap(int slot) { if (on) { long long u = now_ns(); acc[slot] += u - t; t = u; } }
};

// Phase 1 of a query: centre and project q (s.qc, s.qh, s.qa, s.key), derive the per-level tail
// norms (s.a), run the router and rank the cells into s.cells (best hit first; when P exceeds the
// number of hit cells the remaining non-empty cells follow, ordered by level-1 centroid distance).
// stats[1] += M (router hits asked), stats[2] += cells with a hit.
// The cells the router did not hit, ranked by the distance of the query's head to their centroid, appended after
// the hit cells (route_query when P exceeds the hit count; cascade_scan_restart when an iterative scan grows past it).
static void append_rest_cells(Index* ix, Scratch& s) {
    const int d3 = ix->lv[ix->L - 1], d1 = ix->lv[0], C = ix->C;
    std::vector<std::pair<float, int>> rest;
    for (int c = 0; c < C; ++c)
        if (!s.cell_hit[c] && (ix->blk_cnt(c) > 0 || !ix->ov_pos[c].empty()))
            rest.push_back({sq_dist_full(s.qh.data(), &ix->cent[(size_t)c * d3], d1), c});
    std::sort(rest.begin(), rest.end());
    for (auto& r : rest) s.cells.push_back(r);
}

static void route_query(Index* ix, Scratch& s, const float* q, int M, int P, int ef, long long* stats, Prof& pf) {
    const int D = ix->D, L = ix->L, d3 = ix->lv[L - 1], d1 = ix->lv[0], C = ix->C;
    s.qc.resize(D); s.qh.resize(d3); s.qa.resize(d3); s.key.resize(d3);
    float qn2 = 0;
    for (int i = 0; i < D; ++i) { s.qc[i] = q[i] - ix->mean[i]; qn2 += s.qc[i] * s.qc[i]; }
    for (int r = 0; r < d3; ++r) {
        float acc = dot(&ix->proj[(size_t)r * D], s.qc.data(), D);
        s.qh[r] = acc;
        s.qa[r] = acc - ix->lo[r];
        s.key[r] = s.qa[r] / ix->step[r];
    }
    {
        float hn2 = 0; int i = 0;
        for (int l = 0; l < L; ++l) {
            for (; i < ix->lv[l]; ++i) hn2 += s.qh[i] * s.qh[i];
            s.a[l] = std::sqrt(std::max(qn2 - hn2, 0.f));
        }
    }

    s.cell_score.assign(C, INFINITY);
    s.cell_hit.assign(C, 0);
    if (M > 0 && ef >= 0) {  // flat graph router
        if (!ix->flat && ix->rg.dirty) refresh_graph(ix);
        if (ix->bgraph && ix->seed_C > 0 && ix->seed_graph()) {
            // R1c: the seed router (a graph over sub-centroid sketches) ranks cells; the medoids of its top
            // seed_C cells seed the per-point beam
            graph_search(ix, ix->seed_graph(), s, s.key.data(), std::max(ix->seed_ef, ix->seed_C), ix->seed_ef, 0, false, nullptr, 0, false);
            std::vector<std::pair<float, int>> sc_;
            std::vector<char> seen(C, 0);
            for (auto& pr : s.res) {  // best score per cell, in score order
                const int c = ix->seed_cell[pr.second];
                if (!seen[c]) { seen[c] = 1; sc_.push_back({pr.first, c}); }
            }
            std::sort(sc_.begin(), sc_.end());
            std::vector<int> entries;
            for (auto& pr : sc_) {
                if ((int)entries.size() >= ix->seed_C) break;
                const int m = ix->cell_medoid[pr.second];
                if (m >= 0) entries.push_back(m);
            }
            if (entries.empty()) graph_search(ix, ix->bgraph, s, s.key.data(), std::max(ef, M), M, 0, false);
            else graph_search(ix, ix->bgraph, s, s.key.data(), std::max(ef, M), M, 0, false, entries.data(), (int)entries.size());
        } else if (ix->bgraph) graph_search(ix, ix->bgraph, s, s.key.data(), std::max(ef, M), M, 0, false);
        else { s.res.clear(); s.scored.clear(); }
        // frozen index: the points appended since the image live in the extension's delta graph (ids
        // = position - nb); same scoring, so both hit sets rank cells alike. The hits are the top of the
        // UNION of the two searches, as one graph over base + delta would return: appending the delta's
        // own M hits marked up to M extra cells (random far points), which displaced the centroid fill
        // and the weakest base cells and cost 4 pt of recall at 1M with 2,000 live entries (M1, 2026-10-27).
        // Skipped while no appended position is live (after a VACUUM the delta is all tombstones and a
        // bounded walk still costs); otherwise the beam's bound counts tombstoned nodes (dead_bound).
        if (ix->dgraph && ix->dgraph->n > 0 && ix->n_delta_live > 0) {
            const size_t nb_res = s.res.size(), nb_scored = s.scored.size();
            graph_search(ix, ix->dgraph, s, s.key.data(), std::max(ef, M), M, ix->dgraph_nb, true, nullptr, 0, true, true);
            const int keep = ix->router_mode == 0 ? M : std::max(ef, M);
            std::inplace_merge(s.res.begin(), s.res.begin() + nb_res, s.res.end());
            if ((int)s.res.size() > keep) s.res.resize(keep);
            if (ix->router_mode == 2) {  // the delta's visited set is most of the delta: only its merged hits mark cells
                s.scored.resize(nb_scored);
                for (auto& pr : s.res) if (pr.second >= ix->dgraph_nb) s.scored.push_back(pr);
            }
        }
        if (stats) stats[1] += M;
        const auto& hits = ix->router_mode == 2 ? s.scored : s.res;
        for (auto& pr : hits) {
            if (!ix->reduced() && ix->deleted[pr.second]) continue;
            int c = ix->router_cell(pr.second);
            if ((unsigned)c >= (unsigned)C) {
                if (!s.corrupt[0]) snprintf(s.corrupt, sizeof s.corrupt, "corrupt image: cell of position %d is %d, not in [0, %d)", pr.second, c, C);
                continue;
            }
            if (pr.first < s.cell_score[c]) s.cell_score[c] = pr.first;
            s.cell_hit[c] = 1;
        }
    } else if (M > 0 && ix->hnsw) {      // hnswlib router (ef < 0 -> use |ef|; owned indexes only)
        ix->hnsw->setEf(std::max(-ef, M));
        auto res = ix->hnsw->searchKnn(s.key.data(), M);
        if (stats) stats[1] += M;
        while (!res.empty()) {
            auto pr = res.top(); res.pop();
            int c = ix->router_cell(pr.second);
            if (pr.first < s.cell_score[c]) s.cell_score[c] = pr.first;
            s.cell_hit[c] = 1;
        }
    }
    pf.lap(0);
    s.cells.clear();
    int n_hit = 0;
    for (int c = 0; c < C; ++c) if (s.cell_hit[c]) { s.cells.push_back({s.cell_score[c], c}); ++n_hit; }
    std::sort(s.cells.begin(), s.cells.end());
    if (stats) stats[2] += n_hit;  // cells with at least one router hit (the router's coverage)
    if (P > n_hit) append_rest_cells(ix, s);
    pf.lap(5);
}

// Phase 2: the level cascade over cell c under the current k-th distance tau2 with the tail
// cosines ch (ix->ch, or a per-scan calibrated set). Survivors that are not tombstoned land in
// s.cand as (lower bound, pos), sorted by bound; dimension-ops accumulate in s.ops.
static void scan_cell(Index* ix, Scratch& s, int c, float tau2, const float* ch, Prof& pf) {
    const int L = ix->L, d3 = ix->lv[L - 1], d1 = ix->lv[0];
    const float slack = 1.f + 1e-5f;
    const float* a = s.a;
    s.cand.clear();

    // generic cascade over a block: get(j, l) -> pointer to point j's level-l codes
    auto scan_block = [&](int cnt, auto get, auto pos_of_j) {
        s.head_d2.resize(cnt);
        s.alive.clear();
        for (int j = 0; j < cnt; ++j) {
            float hd = sq_dist_codes(s.qa.data(), ix->step.data(), get(j, 0), d1);
            s.head_d2[j] = hd;
            int pos = pos_of_j(j);
            float lb2 = lower_bound2(hd, ix->err_at(0, pos), a[0], ix->tail_at(0, pos), ch[0]);
            if (lb2 <= tau2 * slack) s.alive.push_back(j);
        }
        s.ops += (long long)cnt * d1;
        pf.lap(1);
        for (int l = 1; l < L && !s.alive.empty(); ++l) {
            int lo_d = ix->lv[l - 1], w = ix->lv[l] - lo_d;
            s.alive2.clear();
            const int na = (int)s.alive.size();
            for (int t = 0; t < na; ++t) {
                const int j = s.alive[t];
                // the survivors' codes are a strided subset of the level block: prefetch a few ahead
                // so the scan is bandwidth-bound rather than latency-bound once the block leaves cache
                if (t + PF_AHEAD < na) __builtin_prefetch(get(s.alive[t + PF_AHEAD], l));
                float hd = s.head_d2[j] + sq_dist_codes(s.qa.data() + lo_d, ix->step.data() + lo_d, get(j, l), w);
                s.head_d2[j] = hd;
                int pos = pos_of_j(j);
                float lb2 = lower_bound2(hd, ix->err_at(l, pos), a[l], ix->tail_at(l, pos), ch[l]);
                if (lb2 <= tau2 * slack) s.alive2.push_back(j);
            }
            s.ops += (long long)s.alive.size() * w;
            s.alive.swap(s.alive2);
            pf.lap(l == 1 ? 2 : 3);
        }
        for (int j : s.alive) {
            int pos = pos_of_j(j);
            if (ix->deleted[pos]) continue;
            float lb2 = lower_bound2(s.head_d2[j], ix->err_at(L - 1, pos), a[L - 1], ix->tail_at(L - 1, pos), ch[L - 1]);
            s.cand.push_back({lb2, pos});
        }
    };

    const int cnt = ix->blk_cnt(c);
    if (cnt) {
        const int8_t* base = ix->blk_codes_ptr(c);
        const int* bpos = ix->blk_pos_ptr(c);
        const unsigned tot = (unsigned)ix->total();
        for (int j = 0; j < cnt; ++j)   // P5a: a corrupt blocked position would index the per-point arrays out of range
            if ((unsigned)bpos[j] >= tot) {
                if (!s.corrupt[0]) snprintf(s.corrupt, sizeof s.corrupt, "corrupt image: cell %d holds position %d, not in [0, %u)", c, bpos[j], tot);
                return;
            }
        scan_block(cnt,
                   [&](int j, int l) {
                       int lo_d = l ? ix->lv[l - 1] : 0, w = ix->lv[l] - lo_d;
                       return base + (size_t)cnt * lo_d + (size_t)j * w;
                   },
                   [&](int j) { return bpos[j]; });
    }
    const auto& ovp = ix->ov_pos[c];
    if (!ovp.empty()) {
        const int8_t* obase = ix->ov_codes[c].data();
        scan_block((int)ovp.size(),
                   [&](int j, int l) { return obase + (size_t)j * d3 + (l ? ix->lv[l - 1] : 0); },
                   [&](int j) { return ovp[j]; });
    }
    if (!s.cand.empty()) std::sort(s.cand.begin(), s.cand.end());
}

static void search_one(Index* ix, Scratch& s, const float* q, int k, int M, int P, int ef,
                       int* out_ids, float* out_d2, long long* stats) {
    const int D = ix->D;
    // profile: stats[3] projection+router, [4] level-1 scan, [5] level-2 gather, [6] level-3 gather,
    // [7] full-row reads (exact distances), [8] cell ranking + rest  (nanoseconds, summed over queries)
    Prof pf; pf.on = ix->profile && stats; pf.start();
    s.ops = 0;
    route_query(ix, s, q, M, P, ef, stats, pf);

    std::priority_queue<std::pair<float, int>> heap;
    float tau2 = INFINITY;
    const float slack = 1.f + 1e-5f;
    const int nP = std::min<int>(P, (int)s.cells.size());
    const bool have_x = ix->has_x();  // a deserialised (sketch-only) index: rank by lower bound instead
    for (int ci = 0; ci < nP; ++ci) {
        scan_cell(ix, s, s.cells[ci].second, tau2, ix->ch, pf);
        if (s.cand.empty()) continue;
        pf.lap(5);
        const int nc = (int)s.cand.size();
        auto prefetch_row = [&](int ci_) {  // the full row of a candidate: D*4 bytes, random over the corpus
            const char* r = (const char*)ix->vec(s.cand[ci_].second);
            for (int b = 0; b < D * 4; b += 64) __builtin_prefetch(r + b);
        };
        if (have_x) for (int t = 0; t < std::min(nc, PF_ROWS); ++t) prefetch_row(t);
        for (int t = 0; t < nc; ++t) {
            auto& pr = s.cand[t];
            if (pr.first > tau2 * slack) break;
            float d2 = pr.first;
            if (have_x) {
                if (t + PF_ROWS < nc) prefetch_row(t + PF_ROWS);
                d2 = sq_dist_full(q, ix->vec(pr.second), D);
                s.ops += D;
            }
            if ((int)heap.size() < k) { heap.push({d2, pr.second}); if ((int)heap.size() == k) tau2 = heap.top().first; }
            else if (d2 < heap.top().first) { heap.pop(); heap.push({d2, pr.second}); tau2 = heap.top().first; }
        }
        pf.lap(4);
    }
    if (stats) stats[0] += s.ops;
    if (pf.on) { pf.lap(5); for (int i = 0; i < 6; ++i) stats[3 + i] += pf.acc[i]; }
    int m = (int)heap.size();
    for (int i = 0; i < k; ++i) { out_ids[i] = -1; out_d2[i] = INFINITY; }
    for (int i = m - 1; i >= 0; --i) { out_ids[i] = ix->perm_at(heap.top().second); out_d2[i] = heap.top().first; heap.pop(); }
}

// Rebuild the flat router graph now (otherwise the first search after a build/insert does it).
// Call this before searching from several threads at once.
void cascade_prepare(void* p) {
    Index* ix = (Index*)p;
    if (!ix->flat && ix->rg.dirty) refresh_graph(ix);
}

void cascade_search(void* p, const float* queries, int nq, int k, int M, int P, int ef,
                    int* out_ids, float* out_d2, long long* stats) {
    Index* ix = (Index*)p;
    Scratch s;
    for (int i = 0; i < nq; ++i)
        search_one(ix, s, queries + (size_t)i * ix->D, k, M, P, ef,
                   out_ids + (size_t)i * k, out_d2 + (size_t)i * k, stats);
}

// Experiment T6: replace the per-point router by a graph over n_nodes node sketches (float heads in
// the PCA space, quantised with the index's lo/step) each mapped to a cell. Build with hnswlib
// (M, efC); the flat graph is refreshed lazily.
void cascade_set_router_nodes(void* p, const float* node_heads, const int* node_cells, int n_nodes, int M, int efC) {
    Index* ix = (Index*)p;
    const int d3 = ix->lv[ix->L - 1];
    delete ix->hnsw;
    ix->space->prm.query_float = false;
    ix->hnsw = new hnswlib::HierarchicalNSW<float>(ix->space, (size_t)n_nodes, M, efC);
    ix->rcodes.assign((size_t)n_nodes * d3, 0);
    ix->rnode_cell.assign(node_cells, node_cells + n_nodes);
    ix->n_rnodes = n_nodes;
    std::vector<int8_t> row(d3);
    for (int i = 0; i < n_nodes; ++i) {
        const float* h = node_heads + (size_t)i * d3;
        for (int k = 0; k < d3; ++k) {
            int q = (int)std::lrint((h[k] - ix->lo[k]) / ix->step[k]);
            row[k] = (int8_t)std::max(-128, std::min(127, q));
        }
        std::memcpy(&ix->rcodes[(size_t)i * d3], row.data(), d3);
        ix->hnsw->addPoint(row.data(), (size_t)i);
    }
    ix->space->prm.query_float = true;
    ix->rg.dirty = true;
}

// Which distance kernels this translation unit was compiled with (reported by the PostgreSQL
// extension's trigpoint_version(); the Python side infers it from the build flags).
const char* cascade_kernel() {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    return "neon+dotprod";
#elif defined(__ARM_NEON)
    return "neon";
#elif defined(__AVX2__) && defined(__FMA__)
    return "avx2+fma";
#else
    return "scalar";
#endif
}

void cascade_set_router_mode(void* p, int mode) { ((Index*)p)->router_mode = mode; }
int cascade_get_router_mode(void* p) { return ((Index*)p)->router_mode; }
void cascade_set_profile(void* p, int on) { ((Index*)p)->profile = on; }

// R1a: router scoring width (0 or >= d3: full sketch, today's behaviour)
void cascade_set_router_width(void* p, int w) { ((Index*)p)->router_width = w; }

// R1c: the seed router — a graph over n_nodes node sketches (float PCA heads, as cascade_set_router_nodes)
// mapped to cells, built with hnswlib (M, efC) and flattened; plus the per-cell medoids (the main-block point
// whose dequantised sketch is nearest the cell's head centroid). Owned, per-point indexes only.
void cascade_set_seed_router(void* p, const float* node_heads, const int* node_cells, int n_nodes, int M, int efC) {
    Index* ix = (Index*)p;
    const int d3 = ix->lv[ix->L - 1], C = ix->C;
    delete ix->seed_hnsw;
    ix->space->prm.query_float = false;
    ix->seed_hnsw = new hnswlib::HierarchicalNSW<float>(ix->space, (size_t)n_nodes, M, efC);
    ix->seed_codes.assign((size_t)n_nodes * d3, 0);
    ix->seed_cell.assign(node_cells, node_cells + n_nodes);
    std::vector<int8_t> row(d3);
    for (int i = 0; i < n_nodes; ++i) {
        const float* h = node_heads + (size_t)i * d3;
        for (int k = 0; k < d3; ++k) {
            int q = (int)std::lrint((h[k] - ix->lo[k]) / ix->step[k]);
            row[k] = (int8_t)std::max(-128, std::min(127, q));
        }
        std::memcpy(&ix->seed_codes[(size_t)i * d3], row.data(), d3);
        ix->seed_hnsw->addPoint(row.data(), (size_t)i);
    }
    ix->space->prm.query_float = true;
    // flatten (node ids = labels; hnswlib's internal ids are in arrival order, so int2pos maps)
    auto* h = ix->seed_hnsw;
    std::vector<int> int2pos(n_nodes), pos2int(n_nodes);
    for (int i = 0; i < n_nodes; ++i) { const int pos = (int)h->getExternalLabel((hnswlib::tableint)i); int2pos[i] = pos; pos2int[pos] = i; }
    int64_t blocks = 0;
    for (int i = 0; i < n_nodes; ++i) blocks += std::max(0, h->element_levels_[(size_t)i]);
    FlatGraph hd;
    const uint64_t bytes = fg_layout(&hd, n_nodes, d3, (int)h->M_, (int32_t)std::max<int64_t>(blocks, 1));
    ix->seed_buf.assign(bytes, 0);
    FlatGraph* fg = (FlatGraph*)ix->seed_buf.data();
    *fg = hd;
    fg->n = n_nodes; fg->efC = (int)h->ef_construction_; fg->maxlevel = h->maxlevel_;
    fg->enterpoint = n_nodes ? int2pos[(int)h->enterpoint_node_] : -1;
    fg->seed = 0; fg->mult = 1.0 / std::log(std::max(2, (int)h->M_));
    const float* w = ix->space->step2.data();
    std::memcpy(fg_w2(fg), w, sizeof(float) * d3);
    std::memcpy(fg_codes(fg), ix->seed_codes.data(), (size_t)n_nodes * d3);
    float* sc = fg_sc(fg);
    for (int pos = 0; pos < n_nodes; ++pos) {
        const int8_t* c = fg_code(fg, pos);
        float acc = 0;
        for (int i = 0; i < d3; ++i) acc += w[i] * (float)c[i] * (float)c[i];
        sc[pos] = acc;
    }
    int32_t* lvl = fg_level(fg); int32_t* upidx = fg_upidx(fg);
    int32_t next_block = 0;
    for (int pos = 0; pos < n_nodes; ++pos) {
        const hnswlib::tableint ii = (hnswlib::tableint)pos2int[pos];
        const int L = h->element_levels_[(size_t)ii];
        lvl[pos] = L;
        auto* ll0 = h->get_linklist0(ii);
        int cnt = (int)h->getListCount(ll0);
        auto* data = (hnswlib::tableint*)(ll0 + 1);
        int32_t* dst = fg_links0(fg, pos);
        int k = 0;
        for (int j = 0; j < cnt && k < fg->M0; ++j) dst[1 + k++] = int2pos[(int)data[j]];
        dst[0] = k;
        upidx[pos] = L > 0 ? next_block : -1;
        for (int level = 1; level <= L; ++level) {
            auto* ll = h->get_linklist(ii, level);
            int c2 = (int)h->getListCount(ll);
            auto* d2 = (hnswlib::tableint*)(ll + 1);
            int32_t* blk = (int32_t*)(ix->seed_buf.data() + fg->uplinks_off) + (size_t)next_block * (fg->M + 1);
            int kk = 0;
            for (int j = 0; j < c2 && kk < fg->M; ++j) blk[1 + kk++] = int2pos[(int)d2[j]];
            blk[0] = kk;
            ++next_block;
        }
    }
    fg->up_used = next_block;
    // medoids: per cell, the main-block point whose dequantised sketch is nearest the head centroid
    ix->cell_medoid.assign(C, -1);
    for (int c = 0; c < C; ++c) {
        const int cnt = ix->blk_cnt(c);
        if (!cnt || ix->rcodes.empty()) continue;
        const int* bpos = ix->blk_pos_ptr(c);
        const float* ce = &ix->cent[(size_t)c * d3];
        float best = INFINITY; int bp = -1;
        for (int j = 0; j < cnt; ++j) {
            const int pos = bpos[j];
            if (ix->deleted[pos]) continue;
            const int8_t* code = &ix->rcodes[(size_t)pos * d3];
            float d = 0;
            for (int k = 0; k < d3; ++k) { const float t = ix->lo[k] + (float)code[k] * ix->step[k] - ce[k]; d += t * t; }
            if (d < best) { best = d; bp = pos; }
        }
        ix->cell_medoid[c] = bp;
    }
}

// R1c: seed_C > 0 enables the seeded beam (top-C cells of the seed router, seed_ef hits asked of it); 0 disables
void cascade_set_seed(void* p, int C, int ef) { Index* ix = (Index*)p; ix->seed_C = C; ix->seed_ef = std::max(ef, C); }

// The router's cell ranking for one query (R1 metrics): the first P cells route_query would open, best hit first,
// then the centroid-ranked rest (P = C: the whole order). Returns the count written.
int cascade_route_cells(void* p, const float* q, int M, int P, int ef, int* cells_out) {
    Index* ix = (Index*)p;
    Scratch s;
    long long st[3] = {0, 0, 0};
    Prof pf;
    route_query(ix, s, q, M, P, std::max(ef, M), st, pf);
    const int n = std::min<int>(P, (int)s.cells.size());
    for (int i = 0; i < n; ++i) cells_out[i] = s.cells[i].second;
    return n;
}

void cascade_set_ch(void* p, const float* ch) {
    Index* ix = (Index*)p;
    for (int l = 0; l < ix->L; ++l) ix->ch[l] = ch[l];
}

long long cascade_bytes(void* p) {
    Index* ix = (Index*)p;
    long long ov = 0;
    for (auto& v : ix->ov_codes) ov += (long long)v.size();
    for (auto& v : ix->blk_codes) ov += (long long)v.size();
    return (long long)(ov + 4LL * ix->err[0].size() * 2 * ix->L +
                       4 * ix->cent.size() + 4 * ix->perm.size() + 4 * ix->cell_of.size());
}


// =============================================================================================
// Build from raw vectors, serialisation and a two-phase scan API: the PostgreSQL extension's
// surface (pgext/), also reachable from Python for tests. Everything here is single-threaded and
// self-contained (no LAPACK / faiss): PCA head by covariance + block subspace iteration, k-means on
// the heads for the cells, cascade_build for codes / norms / router, then the conformal
// calibration from exact searches over the freshly built index. Long phases poll the abort hook.
// =============================================================================================

}  // extern "C"  (the helpers below are C++: templates, lambdas)

static const int CAL_K = 10;  // k of the calibration pairs (the project's k)

struct Rng {  // splitmix64: deterministic across platforms
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    int below(int n) { return (int)(next() % (uint64_t)n); }
    double uniform() { return (double)(next() >> 11) * (1.0 / 9007199254740992.0); }
    double normal() {
        double u = uniform(), v = uniform();
        return std::sqrt(-2.0 * std::log(u + 1e-300)) * std::cos(6.283185307179586 * v);
    }
};

// m distinct indices out of n, sorted
static std::vector<int> sample_without_replacement(int n, int m, Rng& rng) {
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    for (int i = 0; i < m; ++i) std::swap(idx[i], idx[i + rng.below(n - i)]);
    idx.resize(m);
    std::sort(idx.begin(), idx.end());
    return idx;
}

static inline double ddot(const double* a, const double* b, int n) {
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int i = 0;
    for (; i + 4 <= n; i += 4) { s0 += a[i] * b[i]; s1 += a[i + 1] * b[i + 1]; s2 += a[i + 2] * b[i + 2]; s3 += a[i + 3] * b[i + 3]; }
    for (; i < n; ++i) s0 += a[i] * b[i];
    return (s0 + s1) + (s2 + s3);
}

// modified Gram-Schmidt on the r columns (each D contiguous doubles) of Q
static void orthonormalise(std::vector<double>& Q, int D, int r) {
    for (int j = 0; j < r; ++j) {
        double* qj = &Q[(size_t)j * D];
        for (int pass = 0; pass < 2; ++pass)
            for (int i = 0; i < j; ++i) {
                const double* qi = &Q[(size_t)i * D];
                double d = ddot(qi, qj, D);
                for (int k = 0; k < D; ++k) qj[k] -= d * qi[k];
            }
        double nrm = std::sqrt(ddot(qj, qj, D));
        if (nrm < 1e-300) nrm = 1;
        for (int k = 0; k < D; ++k) qj[k] /= nrm;
    }
}

// cyclic Jacobi on the symmetric n x n matrix A (row-major, destroyed): eigenvalues w, eigenvectors
// as the columns of V (row-major n x n, V[k*n + j] = component k of eigenvector j)
static void jacobi_eigen(std::vector<double>& A, int n, std::vector<double>& V, std::vector<double>& w) {
    V.assign((size_t)n * n, 0.0);
    for (int i = 0; i < n; ++i) V[(size_t)i * n + i] = 1.0;
    double scale = 0;
    for (int i = 0; i < n; ++i) scale += std::fabs(A[(size_t)i * n + i]);
    scale = std::max(scale, 1e-300);
    for (int sweep = 0; sweep < 60; ++sweep) {
        double off = 0;
        for (int i = 0; i < n; ++i) for (int j = i + 1; j < n; ++j) off += A[(size_t)i * n + j] * A[(size_t)i * n + j];
        if (std::sqrt(off) < 1e-13 * scale) break;
        for (int p = 0; p < n; ++p)
            for (int q = p + 1; q < n; ++q) {
                double apq = A[(size_t)p * n + q];
                if (std::fabs(apq) < 1e-300) continue;
                double theta = (A[(size_t)q * n + q] - A[(size_t)p * n + p]) / (2 * apq);
                double t = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1));
                double c = 1 / std::sqrt(t * t + 1), s = t * c;
                for (int k = 0; k < n; ++k) {  // A <- A J  (columns p, q)
                    double akp = A[(size_t)k * n + p], akq = A[(size_t)k * n + q];
                    A[(size_t)k * n + p] = c * akp - s * akq; A[(size_t)k * n + q] = s * akp + c * akq;
                }
                for (int k = 0; k < n; ++k) {  // A <- J^T A  (rows p, q)
                    double apk = A[(size_t)p * n + k], aqk = A[(size_t)q * n + k];
                    A[(size_t)p * n + k] = c * apk - s * aqk; A[(size_t)q * n + k] = s * apk + c * aqk;
                }
                for (int k = 0; k < n; ++k) {  // V <- V J
                    double vkp = V[(size_t)k * n + p], vkq = V[(size_t)k * n + q];
                    V[(size_t)k * n + p] = c * vkp - s * vkq; V[(size_t)k * n + q] = s * vkp + c * vkq;
                }
            }
    }
    w.resize(n);
    for (int i = 0; i < n; ++i) w[i] = A[(size_t)i * n + i];
}

// PCA head of the rows x[idx]: mean (float, as the index uses it), the D x D covariance in double
// (256-row panels, transposed, 4-row register blocking over float dot products), then block
// subspace iteration with d3 + 64 vectors and a Rayleigh-Ritz step for the ordering. proj: d3 x D
// orthonormal rows, descending variance. Returns the share of variance captured (-1 if aborted).
static float fit_head(const float* x, const std::vector<int>& idx, int D, int d3, int iters, Rng& rng,
                      float* proj, float* mean) {
    const int m = (int)idx.size();
    {
        std::vector<double> mu(D, 0.0);
        for (int i : idx) { const float* r = x + (size_t)i * D; for (int k = 0; k < D; ++k) mu[k] += r[k]; }
        for (int k = 0; k < D; ++k) mean[k] = (float)(mu[k] / std::max(m, 1));
    }
    std::vector<double> cov((size_t)D * D, 0.0);
    const int B = 256;
    const int npanels = (m + B - 1) / B;
    {
        const int T = std::max(1, g_build_threads);
        std::vector<std::vector<double>> part(T);
        parallel_for(npanels, [&](int t, int p_lo, int p_hi) {
            std::vector<double>& cv = part[t];
            cv.assign((size_t)D * D, 0.0);
            std::vector<float> panel((size_t)D * B, 0.f);
            for (int pn = p_lo; pn < p_hi; ++pn) {
                const int s0 = pn * B, nb = std::min(B, m - s0);
                for (int k = 0; k < D; ++k) std::fill(&panel[(size_t)k * B], &panel[(size_t)k * B] + B, 0.f);
                for (int b = 0; b < nb; ++b) {
                    const float* r = x + (size_t)idx[s0 + b] * D;
                    for (int k = 0; k < D; ++k) panel[(size_t)k * B + b] = r[k] - mean[k];
                }
                for (int i = 0; i < D; i += 4) {
                    const int ni = std::min(4, D - i);
                    const float* pi[4];
                    for (int a = 0; a < ni; ++a) pi[a] = &panel[(size_t)(i + a) * B];
                    for (int j = i; j < D; ++j) {
                        const float* pj = &panel[(size_t)j * B];
                        for (int a = 0; a < ni; ++a)
                            if (j >= i + a) cv[(size_t)(i + a) * D + j] += dot(pi[a], pj, B);
                    }
                }
            }
        });
        for (auto& cv : part) if (!cv.empty()) for (size_t i = 0; i < cov.size(); ++i) cov[i] += cv[i];
    }
    if (g_abort_fn && g_abort_fn(g_abort_ctx)) return -1.f;
    const double denom = std::max(m - 1, 1);
    double trace = 0;
    for (int i = 0; i < D; ++i) {
        for (int j = i; j < D; ++j) { double v = cov[(size_t)i * D + j] / denom; cov[(size_t)i * D + j] = v; cov[(size_t)j * D + i] = v; }
        trace += cov[(size_t)i * D + i];
    }
    const int r = std::min(D, d3 + 64);
    std::vector<double> Q((size_t)r * D), Y((size_t)r * D);  // column j = Q[j*D .. j*D+D)
    for (auto& v : Q) v = rng.normal();
    orthonormalise(Q, D, r);
    auto mul = [&](const std::vector<double>& src, std::vector<double>& dst) {  // dst = cov * src, per column
        parallel_for(r, [&](int, int j_lo, int j_hi) {
            for (int j = j_lo; j < j_hi; ++j) {
                const double* qj = &src[(size_t)j * D];
                double* yj = &dst[(size_t)j * D];
                for (int k = 0; k < D; ++k) yj[k] = ddot(&cov[(size_t)k * D], qj, D);
            }
        });
    };
    for (int it = 0; it < iters; ++it) {
        if (g_abort_fn && g_abort_fn(g_abort_ctx)) return -1.f;
        mul(Q, Y);
        Q.swap(Y);
        orthonormalise(Q, D, r);
    }
    mul(Q, Y);
    std::vector<double> Bm((size_t)r * r), U, w;
    for (int a = 0; a < r; ++a)
        for (int b = a; b < r; ++b) { double v = ddot(&Q[(size_t)a * D], &Y[(size_t)b * D], D); Bm[(size_t)a * r + b] = v; Bm[(size_t)b * r + a] = v; }
    jacobi_eigen(Bm, r, U, w);
    std::vector<int> order(r);
    for (int i = 0; i < r; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return w[a] > w[b]; });
    double captured = 0;
    for (int t = 0; t < d3; ++t) {
        const int e = order[t];
        captured += std::max(w[e], 0.0);
        for (int k = 0; k < D; ++k) {
            double v = 0;
            for (int a = 0; a < r; ++a) v += Q[(size_t)a * D + k] * U[(size_t)a * r + e];
            proj[(size_t)t * D + k] = (float)v;
        }
    }
    return trace > 0 ? (float)(captured / trace) : 0.f;
}

// Lloyd k-means on the d-dim heads: trained on at most train_cap rows (random init from the
// training rows, empty clusters re-seeded), then every row assigned. Returns false if aborted.
static bool kmeans_heads(const float* H, int n, int d, int C, int iters, int train_cap, Rng& rng,
                         float* cent, int* labels) {
    std::vector<int> tr;
    if (n > train_cap) tr = sample_without_replacement(n, train_cap, rng);
    else { tr.resize(n); for (int i = 0; i < n; ++i) tr[i] = i; }
    const int T = (int)tr.size();
    {
        std::vector<int> init = sample_without_replacement(T, std::min(C, T), rng);
        for (int c = 0; c < C; ++c) {
            const int src = c < (int)init.size() ? tr[init[c]] : tr[rng.below(T)];
            std::memcpy(cent + (size_t)c * d, H + (size_t)src * d, sizeof(float) * d);
        }
    }
    std::vector<float> cn2(C);
    std::vector<double> sum((size_t)C * d);
    std::vector<int> cnt(C), lab(T);
    auto nearest = [&](const float* h) {
        int best = 0; float bd = INFINITY;
        for (int c = 0; c < C; ++c) {
            float v = cn2[c] - 2.f * dot(h, cent + (size_t)c * d, d);
            if (v < bd) { bd = v; best = c; }
        }
        return best;
    };
    auto norms = [&]() { for (int c = 0; c < C; ++c) cn2[c] = dot(cent + (size_t)c * d, cent + (size_t)c * d, d); };
    const int NT = std::max(1, g_build_threads);
    std::vector<std::vector<double>> psum(NT); std::vector<std::vector<int>> pcnt(NT);
    for (int it = 0; it < iters; ++it) {
        if (g_abort_fn && g_abort_fn(g_abort_ctx)) return false;
        norms();
        parallel_for(T, [&](int th, int lo, int hi) {
            std::vector<double>& ps = psum[th]; std::vector<int>& pc = pcnt[th];
            ps.assign((size_t)C * d, 0.0); pc.assign(C, 0);
            for (int t = lo; t < hi; ++t) {
                const float* h = H + (size_t)tr[t] * d;
                int c = nearest(h); lab[t] = c; pc[c]++;
                double* sc = &ps[(size_t)c * d];
                for (int k = 0; k < d; ++k) sc[k] += h[k];
            }
        });
        std::fill(sum.begin(), sum.end(), 0.0); std::fill(cnt.begin(), cnt.end(), 0);
        for (int th = 0; th < NT; ++th) {
            if (psum[th].empty()) continue;
            for (size_t i = 0; i < sum.size(); ++i) sum[i] += psum[th][i];
            for (int c = 0; c < C; ++c) cnt[c] += pcnt[th][c];
        }
        for (int c = 0; c < C; ++c) {
            if (cnt[c] == 0) { std::memcpy(cent + (size_t)c * d, H + (size_t)tr[rng.below(T)] * d, sizeof(float) * d); continue; }
            for (int k = 0; k < d; ++k) cent[(size_t)c * d + k] = (float)(sum[(size_t)c * d + k] / cnt[c]);
        }
    }
    norms();
    if (g_abort_fn && g_abort_fn(g_abort_ctx)) return false;
    parallel_for(n, [&](int, int lo, int hi) { for (int i = lo; i < hi; ++i) labels[i] = nearest(H + (size_t)i * d); });
    return true;
}

static float ch_for_delta(const std::vector<float>& g, double delta) {
    if (g.empty()) return 1.f;
    const double N = (double)g.size();
    for (int i = 0; i <= 400; ++i) {  // the Python grid: np.linspace(-1, 1, 401)
        const float gv = -1.f + 0.005f * (float)i;
        const double above = (double)(g.end() - std::upper_bound(g.begin(), g.end(), gv));
        if (above / N <= delta) return gv;
    }
    return 1.f;
}

// Conformal calibration (run_q2.calibrate in C++): n_cal corpus points as queries, their exact
// 10-NN from the index itself in exact mode (c_h = 1, all cells), and per level the critical
// cosine g* of every (query, neighbour) pair above which the calibrated bound would miss it:
// lb^2 > tau^2  <=>  g < (head_lb^2 + a^2 + b^2 - tau^2) / (2ab). Stores the sorted g* per level
// and sets ix->ch to c_h(delta). Needs the full vectors (build time). Returns false if aborted.
static bool calibrate_index(Index* ix, const float* H, int n_cal, double delta, Rng& rng) {
    const int n = ix->n0, L = ix->L, d3 = ix->lv[L - 1];
    std::vector<int> ts = sample_without_replacement(n, std::min(n_cal, n), rng);
    float ch_saved[4];
    for (int l = 0; l < L; ++l) { ch_saved[l] = ix->ch[l]; ix->ch[l] = 1.f; ix->cal[l].clear(); }
    if (ix->rg.dirty) refresh_graph(ix);  // before the (read-only, concurrent) searches
    if (g_abort_fn && g_abort_fn(g_abort_ctx)) { for (int l = 0; l < L; ++l) ix->ch[l] = ch_saved[l]; return false; }
    const int NT = std::max(1, g_build_threads);
    std::vector<std::vector<float>> pcal((size_t)NT * 4);
    parallel_for((int)ts.size(), [&](int th, int lo, int hi) {
        Scratch s;
        std::vector<int> ids(CAL_K + 1), kept;
        std::vector<float> d2(CAL_K + 1), qa(d3);
        for (int qi = lo; qi < hi; ++qi) {
            const int t = ts[qi];
            const int pos_t = ix->pos_of[t];
            search_one(ix, s, ix->vec(pos_t), CAL_K + 1, 100, ix->C, 100, ids.data(), d2.data(), nullptr);
            kept.clear();
            float tau2 = 0;
            for (int i = 0; i <= CAL_K && (int)kept.size() < CAL_K; ++i)
                if (ids[i] >= 0 && ids[i] != t) { kept.push_back(ids[i]); tau2 = d2[i]; }
            if ((int)kept.size() < CAL_K) continue;
            const float* ht = H + (size_t)t * d3;
            for (int i = 0; i < d3; ++i) qa[i] = ht[i] - ix->lo[i];
            for (int l = 0; l < L; ++l) {
                const int hl = ix->lv[l];
                const double a = ix->tail[l][pos_t];
                for (int r : kept) {
                    const int pos_r = ix->pos_of[r];
                    const double hd = sq_dist_codes(qa.data(), ix->step.data(), &ix->rcodes[(size_t)pos_r * d3], hl);
                    double hlb = std::sqrt(std::max(hd, 0.0)) - ix->err[l][pos_r];
                    if (hlb < 0) hlb = 0;
                    const double b = ix->tail[l][pos_r];
                    const double num = hlb * hlb + a * a + b * b - (double)tau2, den = 2 * a * b;
                    double g = den > 0 ? num / den : (num > 0 ? 2.0 : -2.0);
                    pcal[(size_t)th * 4 + l].push_back((float)std::max(-2.0, std::min(2.0, g)));
                }
            }
        }
    });
    for (int l = 0; l < L; ++l) {
        for (int th = 0; th < NT; ++th) ix->cal[l].insert(ix->cal[l].end(), pcal[(size_t)th * 4 + l].begin(), pcal[(size_t)th * 4 + l].end());
        std::sort(ix->cal[l].begin(), ix->cal[l].end());
        ix->ch[l] = ch_for_delta(ix->cal[l], delta);
    }
    return true;
}

static void set_err(char* err, int errlen, const char* msg) {
    if (err && errlen > 0) { std::strncpy(err, msg, (size_t)errlen - 1); err[errlen - 1] = 0; }
}

namespace {
struct Writer {
    std::string buf;
    template <class T> void pod(const T& v) { buf.append((const char*)&v, sizeof v); }
    template <class T> void vec(const std::vector<T>& v) { uint64_t n = v.size(); pod(n); if (n) buf.append((const char*)v.data(), n * sizeof(T)); }
};
struct Reader {
    const char* p; const char* end; bool ok = true;
    Reader(const void* b, size_t n) : p((const char*)b), end((const char*)b + n) {}
    bool take(void* dst, size_t n) { if (!ok || (size_t)(end - p) < n) { ok = false; return false; } std::memcpy(dst, p, n); p += n; return true; }
    template <class T> bool pod(T& v) { return take(&v, sizeof v); }
    template <class T> bool vec(std::vector<T>& v) {
        uint64_t n = 0; if (!pod(n)) return false;
        if (n > (uint64_t)(end - p) / sizeof(T)) { ok = false; return false; }
        v.resize((size_t)n); return n ? take(v.data(), (size_t)n * sizeof(T)) : true;
    }
};
struct FlatLayout {  // offset bookkeeping: every array starts 64-byte aligned
    uint64_t off = 0;
    uint64_t reserve(uint64_t bytes) { off = (off + 63) & ~uint64_t(63); uint64_t at = off; off += bytes; return at; }
};
template <class T> inline void flat_put(char* base, uint64_t off, const T* src, size_t n) { if (n) std::memcpy(base + off, src, n * sizeof(T)); }
}  // namespace

extern "C" {

// PCA head alone (tests: compare with numpy's eigh). Returns the share of variance captured.
float cascade_fit_head(const float* x, int n, int D, int d3, int iters, unsigned seed, float* proj, float* mean) {
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    Rng rng(seed);
    return fit_head(x, idx, D, d3, iters, rng, proj, mean);
}

// The whole build from raw vectors. prm = {C, L, lv0, lv1, lv2, lv3, M, efC, sample, kmeans_iters,
// kmeans_train, n_cal, power_iters, seed, threads}; stats_out (may be null) = {t_fit, t_head, t_kmeans,
// t_index, t_calib, var_captured} in seconds. The vectors are only borrowed during the build:
// the returned index is sketch-only (has_x() false) and is searched through the scan API.
// Returns null with a message in err on failure or abort.
void* cascade_build_vectors(const float* x, int n, int D, const int* prm, double* stats_out,
                            abort_fn_t abort_fn, void* abort_ctx, char* err, int errlen) {
    const int C = prm[0], L = prm[1], M = prm[6], efC = prm[7], sample = prm[8], km_iters = prm[9];
    const int km_train = prm[10], n_cal = prm[11], power_iters = prm[12];
    const unsigned seed = (unsigned)prm[13];
    g_build_threads = std::max(1, std::min(64, prm[14]));  // prm has 15 entries since S4 (threads)
    int lv[4] = {prm[2], prm[3], prm[4], prm[5]};
    if (n < 1 || D < 1 || C < 1 || L < 1 || L > 4) { set_err(err, errlen, "bad n / D / cells / levels"); return nullptr; }
    for (int l = 0; l < L; ++l)
        if (lv[l] < 1 || lv[l] > D || (l && lv[l] <= lv[l - 1])) { set_err(err, errlen, "levels must be increasing and <= the dimension"); return nullptr; }
    if (C > n) { set_err(err, errlen, "more cells than rows"); return nullptr; }
    const int d3 = lv[L - 1];
    g_abort_fn = abort_fn; g_abort_ctx = abort_ctx;
    Rng rng(seed);
    auto tick = [] { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    double t0 = tick();
    std::vector<float> proj((size_t)d3 * D), mean(D);
    std::vector<int> idx;
    if (n > sample) idx = sample_without_replacement(n, sample, rng);
    else { idx.resize(n); for (int i = 0; i < n; ++i) idx[i] = i; }
    const float var = fit_head(x, idx, D, d3, power_iters, rng, proj.data(), mean.data());
    if (var < 0) { g_abort_fn = nullptr; set_err(err, errlen, "cancelled"); return nullptr; }
    double t1 = tick();
    std::vector<float> H((size_t)n * d3);
    parallel_for(n, [&](int, int lo, int hi) {
        std::vector<float> xc(D);
        for (int i = lo; i < hi; ++i) {
            const float* r = x + (size_t)i * D;
            for (int k = 0; k < D; ++k) xc[k] = r[k] - mean[k];
            for (int t = 0; t < d3; ++t) H[(size_t)i * d3 + t] = dot(&proj[(size_t)t * D], xc.data(), D);
        }
    });
    if (abort_fn && abort_fn(abort_ctx)) { g_abort_fn = nullptr; g_build_threads = 1; set_err(err, errlen, "cancelled"); return nullptr; }
    std::vector<float> lo(d3), step(d3);
    for (int t = 0; t < d3; ++t) {
        float mn = INFINITY, mx = -INFINITY;
        for (int i = 0; i < n; ++i) { float v = H[(size_t)i * d3 + t]; mn = std::min(mn, v); mx = std::max(mx, v); }
        step[t] = mx > mn ? (mx - mn) / 255.f : 1.f;
        lo[t] = mn + 128.f * step[t];  // signed int8 codes
    }
    double t2 = tick();
    std::vector<float> cent((size_t)C * d3);
    std::vector<int> labels(n);
    if (!kmeans_heads(H.data(), n, d3, C, km_iters, km_train, rng, cent.data(), labels.data())) {
        g_abort_fn = nullptr; set_err(err, errlen, "cancelled"); return nullptr;
    }
    double t3 = tick();
    const float ones[4] = {1, 1, 1, 1};
    Index* ix = (Index*)cascade_build(n, D, C, L, lv, labels.data(), x, H.data(), proj.data(), mean.data(),
                                      lo.data(), step.data(), ones, M, efC, 0, 1);
    if (!ix) { g_abort_fn = nullptr; set_err(err, errlen, "cancelled"); return nullptr; }
    ix->var_captured = var;
    double t4 = tick();
    if (n_cal > 0 && !calibrate_index(ix, H.data(), n_cal, 0.01, rng)) {
        delete ix; g_abort_fn = nullptr; set_err(err, errlen, "cancelled"); return nullptr;
    }
    double t5 = tick();
    ix->xext = nullptr;  // sketch-only from here on: the caller owns (and may free) the vectors
    g_abort_fn = nullptr; g_abort_ctx = nullptr; g_build_threads = 1;
    if (stats_out) { stats_out[0] = t1 - t0; stats_out[1] = t2 - t1; stats_out[2] = t3 - t2; stats_out[3] = t4 - t3; stats_out[4] = t5 - t4; stats_out[5] = var; }
    return ix;
}

// H14: the model of an index built over no rows (CREATE INDEX on an empty table, an UNLOGGED table's init
// fork): the projection keeps the first d3 coordinates (an orthonormal projection, so the level bound of
// Lemma 1 holds with the tail = the norm of the other coordinates), the quantiser covers [-4, 4) in 1/32
// steps (anything beyond saturates and lands in the exact error term), one cell, no calibration (every
// scan uses the exact bound). Inserts then go through the log as usual; REINDEX refits once rows exist.
void* cascade_build_trivial(int D, int L, const int* lv, int M, int efC, char* err, int errlen) {
    if (D < 1 || L < 1 || L > 4) { set_err(err, errlen, "bad D / levels"); return nullptr; }
    for (int l = 0; l < L; ++l)
        if (lv[l] < 1 || lv[l] > D || (l && lv[l] <= lv[l - 1])) { set_err(err, errlen, "levels must be increasing and <= the dimension"); return nullptr; }
    const int d3 = lv[L - 1];
    std::vector<float> proj((size_t)d3 * D, 0.f), mean(D, 0.f), lo(d3, -4.f), step(d3, 1.f / 32.f);
    for (int i = 0; i < d3; ++i) proj[(size_t)i * D + i] = 1.f;
    const float ones[4] = {1, 1, 1, 1};
    Index* ix = (Index*)cascade_build(0, D, 1, L, lv, nullptr, nullptr, nullptr, proj.data(), mean.data(),
                                      lo.data(), step.data(), ones, M, efC, 1, 1);
    if (!ix) { set_err(err, errlen, "trivial build failed"); return nullptr; }
    ix->var_captured = 0.f;
    ix->xext = nullptr;
    return ix;
}

// c_h for a miss probability delta from the stored calibration (1 = exact when none). Returns L.
int cascade_ch_for_delta(void* p, double delta, float* out) {
    Index* ix = (Index*)p;
    for (int l = 0; l < ix->L; ++l) out[l] = ix->cal[l].empty() ? 1.f : ch_for_delta(ix->cal[l], delta);
    return ix->L;
}

float cascade_sq_dist(const float* a, const float* b, int D) { return sq_dist_full(a, b, D); }
float cascade_dist_i8(const int8_t* a, const int8_t* b, const float* w2, int d) { return dist_i8_i8(a, b, w2, d); }

// ---- serialisation (sketch-only: the full vectors are never stored; a router over reduced nodes is not) ----

static const uint32_t TPIX_MAGIC = 0x58495054u;  // "TPIX"
static const uint32_t TPIX_VERSION = 1;

int cascade_serialize(void* p, void** out, size_t* out_len) {
    Index* ix = (Index*)p;
    if (ix->reduced() || !ix->hnsw || ix->flat) return -1;
    Writer w;
    w.pod(TPIX_MAGIC); w.pod(TPIX_VERSION);
    w.pod(ix->n0); w.pod(ix->D); w.pod(ix->C); w.pod(ix->L);
    for (int l = 0; l < 4; ++l) w.pod(ix->lv[l]);
    for (int l = 0; l < 4; ++l) w.pod(ix->ch[l]);
    w.pod(ix->n_inserted); w.pod(ix->n_deleted); w.pod(ix->n_compacted);
    w.pod(ix->router_mode); w.pod(ix->var_captured);
    w.vec(ix->lo); w.vec(ix->step); w.vec(ix->proj); w.vec(ix->mean); w.vec(ix->cent);
    w.vec(ix->perm); w.vec(ix->pos_of); w.vec(ix->cell_of); w.vec(ix->deleted);
    for (int l = 0; l < ix->L; ++l) { w.vec(ix->err[l]); w.vec(ix->tail[l]); w.vec(ix->cal[l]); }
    w.vec(ix->rcodes);
    for (int c = 0; c < ix->C; ++c) { w.vec(ix->blk_codes[c]); w.vec(ix->blk_pos[c]); w.vec(ix->ov_codes[c]); w.vec(ix->ov_pos[c]); }
    std::ostringstream os(std::ios::binary);
    auto st = ix->hnsw->saveIndexNoExceptions(os);
    if (!st.ok()) return -2;
    std::string g = os.str();
    uint64_t gl = g.size(); w.pod(gl); w.buf.append(g);
    void* buf = std::malloc(w.buf.size());
    if (!buf) return -3;
    std::memcpy(buf, w.buf.data(), w.buf.size());
    *out = buf; *out_len = w.buf.size();
    return 0;
}

void cascade_free_buffer(void* buf) { std::free(buf); }

void* cascade_deserialize(const void* buf, size_t len, char* err, int errlen) {
    Reader r(buf, len);
    uint32_t magic = 0, ver = 0;
    r.pod(magic); r.pod(ver);
    if (!r.ok || magic != TPIX_MAGIC || ver != TPIX_VERSION) { set_err(err, errlen, "not a Trigpoint index image (magic / version)"); return nullptr; }
    Index* ix = new Index();
    r.pod(ix->n0); r.pod(ix->D); r.pod(ix->C); r.pod(ix->L);
    for (int l = 0; l < 4; ++l) r.pod(ix->lv[l]);
    for (int l = 0; l < 4; ++l) r.pod(ix->ch[l]);
    r.pod(ix->n_inserted); r.pod(ix->n_deleted); r.pod(ix->n_compacted);
    r.pod(ix->router_mode); r.pod(ix->var_captured);
    if (!r.ok || ix->L < 1 || ix->L > 4 || ix->C < 1) { delete ix; set_err(err, errlen, "corrupt index image (header)"); return nullptr; }
    r.vec(ix->lo); r.vec(ix->step); r.vec(ix->proj); r.vec(ix->mean); r.vec(ix->cent);
    r.vec(ix->perm); r.vec(ix->pos_of); r.vec(ix->cell_of); r.vec(ix->deleted);
    for (int l = 0; l < ix->L; ++l) { r.vec(ix->err[l]); r.vec(ix->tail[l]); r.vec(ix->cal[l]); }
    r.vec(ix->rcodes);
    ix->blk_codes.assign(ix->C, {}); ix->blk_pos.assign(ix->C, {}); ix->ov_codes.assign(ix->C, {}); ix->ov_pos.assign(ix->C, {});
    for (int c = 0; c < ix->C && r.ok; ++c) { r.vec(ix->blk_codes[c]); r.vec(ix->blk_pos[c]); r.vec(ix->ov_codes[c]); r.vec(ix->ov_pos[c]); }
    uint64_t gl = 0; r.pod(gl);
    if (!r.ok || gl > (uint64_t)(r.end - r.p)) { delete ix; set_err(err, errlen, "corrupt index image (body)"); return nullptr; }
    const int d3 = ix->lv[ix->L - 1];
    if ((int)ix->lo.size() != d3 || (int)ix->step.size() != d3 || ix->proj.size() != (size_t)d3 * ix->D ||
        (int)ix->mean.size() != ix->D || ix->perm.size() != ix->cell_of.size() || ix->perm.size() != ix->deleted.size() ||
        ix->rcodes.size() != ix->perm.size() * (size_t)d3) { delete ix; set_err(err, errlen, "corrupt index image (sizes)"); return nullptr; }
    ix->space = new RouterSpace((size_t)d3, ix->step.data());
    ix->hnsw = new hnswlib::HierarchicalNSW<float>(ix->space);
    {
        std::string g(r.p, (size_t)gl);
        std::istringstream is(g, std::ios::binary);
        auto st = ix->hnsw->loadIndexNoExceptions(is, ix->space, 0);
        if (!st.ok()) { delete ix; set_err(err, errlen, "corrupt index image (router graph)"); return nullptr; }
    }
    if ((int)ix->hnsw->cur_element_count != ix->total()) { delete ix; set_err(err, errlen, "corrupt index image (router size)"); return nullptr; }
    ix->space->prm.query_float = true;
    ix->rg.dirty = true;
    return ix;
}

// ---- flat (frozen) image: see FlatHeader ----

// Flat image writer shared by the owned path (cascade_flatten: the graph is hnswlib's, extracted by
// refresh_graph) and the frozen path (cascade_reflatten: base + this process's delta, the graph built by
// the extension over every position). With buf null (or cap too small) returns the bytes needed,
// otherwise writes and returns the bytes written. Blocks are fully repacked: no overflow blocks, no
// tombstoned rows in the blocks (tombstoned positions keep their slot and their `deleted` flag).
// cal_cap: slots to reserve per calibration table (>= the stored count; the parallel build fills them later).
// newpos (H13): old position -> new position, -1 for a position that leaves the image (tombstoned); the
// image is then written over the n_new surviving positions with id = position (perm and pos_of become the
// identity, no tombstones), so a compaction reclaims every dead row. nullptr keeps every position.
static long long flatten_impl(Index* ix, const FlatGraph* graph, const uint64_t* cal_cap_in, void* buf, size_t cap,
                              const int32_t* newpos = nullptr, int n_new = 0) {
    const int L = ix->L, d3 = ix->lv[L - 1], C = ix->C, n_old = ix->total();
    const int n = newpos ? n_new : n_old;
    if (!graph) return -1;
    if (newpos && graph->n < n_new) return -1;
    // live rows per cell and their row codes, gathered from the base blocks and the overflow blocks
    std::vector<std::vector<int>> keep(C);
    for (int c = 0; c < C; ++c) {
        const int cnt = ix->blk_cnt(c);
        const int* bp = ix->blk_pos_ptr(c);
        for (int j = 0; j < cnt; ++j) if (!ix->deleted[bp[j]]) keep[c].push_back(j);          // index into the main block
        for (size_t j = 0; j < ix->ov_pos[c].size(); ++j) if (!ix->deleted[ix->ov_pos[c][j]]) keep[c].push_back(-(int)j - 1);  // -(j+1): overflow row j
    }
    std::vector<uint64_t> bco(C + 1, 0), bpo(C + 1, 0);
    for (int c = 0; c < C; ++c) { bco[c + 1] = bco[c] + (uint64_t)keep[c].size() * d3; bpo[c + 1] = bpo[c] + keep[c].size(); }
    uint64_t cal_cap[4] = {0, 0, 0, 0};
    for (int l = 0; l < L; ++l) cal_cap[l] = std::max<uint64_t>(ix->cal[l].size(), cal_cap_in ? cal_cap_in[l] : 0);
    const int npos_of = newpos ? n : std::max(ix->fb_npos_of + (int)ix->pos_of.size(), n);

    FlatHeader h{};
    FlatLayout lay;
    lay.reserve(sizeof(FlatHeader));
    h.magic = TPFL_MAGIC; h.version = TPFL_VERSION;
    h.n = n; h.D = ix->D; h.C = C; h.L = L;
    for (int l = 0; l < 4; ++l) { h.lv[l] = ix->lv[l]; h.ch[l] = ix->ch[l]; }
    h.M = graph->M; h.efC = graph->efC;
    h.npos_of = npos_of; h.router_mode = ix->router_mode;
    h.n_inserted = ix->n_inserted; h.n_deleted = ix->n_deleted; h.n_compacted = ix->n_compacted;
    h.var_captured = ix->var_captured;
    h.lo = lay.reserve((uint64_t)d3 * 4); h.step = lay.reserve((uint64_t)d3 * 4);
    h.proj = lay.reserve((uint64_t)d3 * ix->D * 4); h.mean = lay.reserve((uint64_t)ix->D * 4);
    h.cent = lay.reserve((uint64_t)C * d3 * 4);
    for (int l = 0; l < L; ++l) { h.cal_n[l] = ix->cal[l].size(); h.cal_cap[l] = cal_cap[l]; h.cal[l] = lay.reserve(cal_cap[l] * 4); }
    h.perm = lay.reserve((uint64_t)n * 4); h.pos_of = lay.reserve((uint64_t)npos_of * 4);
    h.cell_of = lay.reserve((uint64_t)n * 4); h.deleted = lay.reserve((uint64_t)n);
    for (int l = 0; l < L; ++l) { h.err[l] = lay.reserve((uint64_t)n * 4); h.tail[l] = lay.reserve((uint64_t)n * 4); }
    h.blk_code_off = lay.reserve((uint64_t)(C + 1) * 8); h.blk_pos_off = lay.reserve((uint64_t)(C + 1) * 8);
    h.blk_codes = lay.reserve(bco[C]); h.blk_pos = lay.reserve(bpo[C] * 4);
    h.graph_bytes = graph->total_bytes; h.graph = lay.reserve(h.graph_bytes);
    h.total_bytes = lay.reserve(0);
    if (!buf || cap < h.total_bytes) return (long long)h.total_bytes;

    char* b = (char*)buf;
    std::memset(b, 0, h.total_bytes);
    std::memcpy(b, &h, sizeof h);
    flat_put(b, h.lo, ix->lo.data(), d3); flat_put(b, h.step, ix->step.data(), d3);
    flat_put(b, h.proj, ix->proj.data(), ix->proj.size()); flat_put(b, h.mean, ix->mean.data(), ix->mean.size());
    flat_put(b, h.cent, ix->cent.data(), ix->cent.size());
    for (int l = 0; l < L; ++l) flat_put(b, h.cal[l], ix->cal[l].data(), ix->cal[l].size());
    {
        int32_t* perm = (int32_t*)(b + h.perm); int32_t* cell = (int32_t*)(b + h.cell_of); int32_t* pos_of = (int32_t*)(b + h.pos_of);
        for (int i = 0; i < npos_of; ++i) pos_of[i] = -1;
        for (int pos = 0; pos < n_old; ++pos) {
            const int j = newpos ? newpos[pos] : pos;
            if (j < 0) continue;
            if (j >= n) return -1;
            perm[j] = newpos ? j : ix->perm_at(pos); cell[j] = ix->cell_at(pos);
            if (perm[j] >= 0 && perm[j] < npos_of) pos_of[perm[j]] = j;
            b[h.deleted + j] = newpos ? 0 : (ix->deleted[pos] ? 1 : 0);
            for (int l = 0; l < L; ++l) { ((float*)(b + h.err[l]))[j] = ix->err_at(l, pos); ((float*)(b + h.tail[l]))[j] = ix->tail_at(l, pos); }
        }
    }
    flat_put(b, h.blk_code_off, bco.data(), C + 1); flat_put(b, h.blk_pos_off, bpo.data(), C + 1);
    std::vector<int8_t> row(d3);
    for (int c = 0; c < C; ++c) {
        const int m = (int)keep[c].size();
        int8_t* blk = (int8_t*)(b + h.blk_codes + bco[c]);
        int32_t* bpos = (int32_t*)(b + h.blk_pos) + bpo[c];
        const int cnt = ix->blk_cnt(c);
        const int8_t* base = ix->blk_codes_ptr(c);
        const int* bp = ix->blk_pos_ptr(c);
        for (int j = 0; j < m; ++j) {
            const int k = keep[c][j];
            if (k >= 0) {
                bpos[j] = newpos ? newpos[bp[k]] : bp[k];
                for (int l = 0; l < L; ++l) {
                    const int lo_d = l ? ix->lv[l - 1] : 0, w = ix->lv[l] - lo_d;
                    std::memcpy(row.data() + lo_d, base + (size_t)cnt * lo_d + (size_t)k * w, w);
                }
            } else {
                const int oj = -k - 1;
                bpos[j] = newpos ? newpos[ix->ov_pos[c][oj]] : ix->ov_pos[c][oj];
                std::memcpy(row.data(), ix->ov_codes[c].data() + (size_t)oj * d3, d3);
            }
            for (int l = 0; l < L; ++l) {
                const int lo_d = l ? ix->lv[l - 1] : 0, w = ix->lv[l] - lo_d;
                std::memcpy(blk + (size_t)m * lo_d + (size_t)j * w, row.data() + lo_d, w);
            }
        }
    }
    std::memcpy(b + h.graph, graph, graph->total_bytes);
    return (long long)h.total_bytes;
}

// Owned index -> flat image (the Python package's and the threaded build's path): hnswlib's graph is
// extracted first; cells are repacked by the writer. -1 when not flattenable (frozen, reduced, no graph).
long long cascade_flatten(void* p, void* buf, size_t cap) {
    Index* ix = (Index*)p;
    if (ix->flat || ix->reduced() || !ix->hnsw) return -1;
    if (ix->rg.dirty) refresh_graph(ix);
    return flatten_impl(ix, ix->rg.owned(), nullptr, buf, cap);
}

// Frozen index (base + this process's delta) -> a new flat image whose router is `graph`: a FlatGraph the
// caller built over every position (the compaction copies the base graph and inserts the delta's rows
// with the extension's builder). The delta graph hook is not consulted here.
long long cascade_reflatten(void* p, const void* graph, void* buf, size_t cap) {
    Index* ix = (Index*)p;
    const FlatGraph* g = (const FlatGraph*)graph;
    if (!ix->flat || !g || g->magic != FG_MAGIC || g->n < ix->total()) return -1;
    return flatten_impl(ix, g, nullptr, buf, cap);
}

// H13: the surviving positions in position order: newpos[pos] = rank among the live positions, -1 when
// tombstoned. Returns the live count. The compaction builds its graph and TID table from this map.
int cascade_live_map(void* p, int32_t* newpos) {
    Index* ix = (Index*)p;
    const int n = ix->total();
    int m = 0;
    for (int pos = 0; pos < n; ++pos) newpos[pos] = ix->deleted[pos] ? -1 : m++;
    return m;
}

// The id a position carries (perm) and the position an id sits at (-1 when unknown).
int cascade_pos_id(void* p, int pos) { Index* ix = (Index*)p; return (pos >= 0 && pos < ix->total()) ? ix->perm_at(pos) : -1; }
int cascade_id_pos(void* p, int id) { Index* ix = (Index*)p; return ix->pos_of_id(id); }

// Frozen index -> a renumbered image over the live positions only (newpos from cascade_live_map, graph
// over the new positions). The result has no tombstones and id = position.
long long cascade_reflatten_live(void* p, const void* graph, const int32_t* newpos, int n_new, const uint64_t* cal_cap, void* buf, size_t cap) {
    Index* ix = (Index*)p;
    const FlatGraph* g = (const FlatGraph*)graph;
    if (!ix->flat || !g || g->magic != FG_MAGIC || !newpos) return -1;
    return flatten_impl(ix, g, cal_cap, buf, cap, newpos, n_new);
}

// H16: re-calibration at compaction. The query's exact sketch: its PCA head (d3 floats) and per-level tails
// (the norm of what the first lv[l] head coordinates leave of x - mean), as encode_point computes them.
void cascade_query_sketch(void* p, const float* x, float* head, float* tail4) {
    Index* ix = (Index*)p;
    const int D = ix->D, L = ix->L, d3 = ix->lv[L - 1];
    std::vector<float> xc(D);
    float xn2 = 0;
    for (int i = 0; i < D; ++i) { xc[i] = x[i] - ix->mean[i]; xn2 += xc[i] * xc[i]; }
    for (int j = 0; j < d3; ++j) {
        const float* pr = ix->proj.data() + (size_t)j * D;
        float acc = 0;
        for (int i = 0; i < D; ++i) acc += pr[i] * xc[i];
        head[j] = acc;
    }
    float hn2 = 0; int i = 0;
    for (int l = 0; l < L; ++l) {
        for (; i < ix->lv[l]; ++i) hn2 += head[i] * head[i];
        tail4[l] = std::sqrt(std::max(xn2 - hn2, 0.f));
    }
}

// The critical cosine per level of the pair (query with sketch head / tail4, the row with id `id`) at the
// k-th distance tau2: calibrate_index's formula over the stored codes and norms. out[l] for l < L.
int cascade_calibrate_pair(void* p, const float* head, const float* tail4, int id, float tau2, float* out) {
    Index* ix = (Index*)p;
    const int L = ix->L, d3 = ix->lv[L - 1];
    const int pos_r = ix->pos_of_id(id);
    if (pos_r < 0 || !ix->bgraph) return -1;
    std::vector<float> qa(d3);
    for (int i = 0; i < d3; ++i) qa[i] = head[i] - ix->lo[i];
    for (int l = 0; l < L; ++l) {
        const int hl = ix->lv[l];
        const double a = tail4[l];
        const double hd = sq_dist_codes(qa.data(), ix->step.data(), fg_code(ix->bgraph, pos_r), hl);
        double hlb = std::sqrt(std::max(hd, 0.0)) - ix->err_at(l, pos_r);
        if (hlb < 0) hlb = 0;
        const double b = ix->tail_at(l, pos_r);
        const double num = hlb * hlb + a * a + b * b - (double)tau2, den = 2 * a * b;
        double g = den > 0 ? num / den : (num > 0 ? 2.0 : -2.0);
        out[l] = (float)std::max(-2.0, std::min(2.0, g));
    }
    return L;
}

// Open a flat image in place: the returned index reads the caller's buffer for its base (the buffer
// must outlive the index and may be shared read-only between processes) and keeps inserts and
// tombstones in private memory. The router of the appended positions is the extension's delta graph
// (cascade_set_delta_graph). Not serialisable; cascade_reflatten writes the next image.
// Every section offset and size is checked against the image before anything reads through them
// (csrc/flat_check.h, hardening H3): a corrupt header is an error here, never a wild read later.
int cascade_check_flat(const void* buf, size_t len, int deep, char* err, int errlen) {
    return tpfl_check(buf, len, deep, err, (size_t)errlen);
}

void* cascade_open_flat(const void* buf, size_t len, char* err, int errlen) {
    if (tpfl_check(buf, len, 0, err, (size_t)errlen) < 0) return nullptr;
    const FlatHeader* h = (const FlatHeader*)buf;
    const FlatGraph* g = flat_ptr<FlatGraph>(h, h->graph);
    Index* ix = new Index();
    const int L = h->L, d3 = h->lv[L - 1];
    ix->flat = h; ix->nb = h->n; ix->n0 = h->n; ix->fb_npos_of = h->npos_of;
    ix->D = h->D; ix->C = h->C; ix->L = L;
    for (int l = 0; l < 4; ++l) { ix->lv[l] = h->lv[l]; ix->ch[l] = h->ch[l]; }
    ix->var_captured = h->var_captured; ix->router_mode = h->router_mode;
    ix->n_inserted = h->n_inserted; ix->n_deleted = h->n_deleted; ix->n_compacted = h->n_compacted;
    const float* f;
    f = flat_ptr<float>(h, h->lo); ix->lo.assign(f, f + d3);
    f = flat_ptr<float>(h, h->step); ix->step.assign(f, f + d3);
    f = flat_ptr<float>(h, h->proj); ix->proj.assign(f, f + (size_t)d3 * h->D);
    f = flat_ptr<float>(h, h->mean); ix->mean.assign(f, f + h->D);
    f = flat_ptr<float>(h, h->cent); ix->cent.assign(f, f + (size_t)h->C * d3);
    for (int l = 0; l < L; ++l) { f = flat_ptr<float>(h, h->cal[l]); ix->cal[l].assign(f, f + h->cal_n[l]); }
    ix->fb_perm = flat_ptr<int32_t>(h, h->perm); ix->fb_pos_of = flat_ptr<int32_t>(h, h->pos_of);
    ix->fb_cell_of = flat_ptr<int32_t>(h, h->cell_of);
    for (int l = 0; l < L; ++l) { ix->fb_err[l] = flat_ptr<float>(h, h->err[l]); ix->fb_tail[l] = flat_ptr<float>(h, h->tail[l]); }
    ix->fb_bco = flat_ptr<uint64_t>(h, h->blk_code_off); ix->fb_bpo = flat_ptr<uint64_t>(h, h->blk_pos_off);
    ix->fb_bcodes = flat_ptr<int8_t>(h, h->blk_codes); ix->fb_blk_pos = flat_ptr<int32_t>(h, h->blk_pos);
    { const uint8_t* d = flat_ptr<uint8_t>(h, h->deleted); ix->deleted.assign(d, d + h->n); }
    ix->ov_codes.assign(h->C, {}); ix->ov_pos.assign(h->C, {});
    ix->space = new RouterSpace((size_t)d3, ix->step.data());
    ix->space->prm.query_float = true;
    ix->bgraph = g;
    ix->dgraph = nullptr; ix->dgraph_nb = h->n;
    ix->rg.dirty = false;
    return ix;
}

// The extension's router over the positions appended since the image (a FlatGraph whose id i stands
// for position nb + i); null to detach. The pointer may change whenever the extension grows the graph.
void cascade_set_delta_graph(void* p, const void* graph) {
    Index* ix = (Index*)p;
    ix->dgraph = (const FlatGraph*)graph;
    ix->dgraph_nb = ix->nb;
}

// Router parameters the extension needs to build a graph compatible with this index: out = {d3, M, efC};
// *step = the quantisation steps (d3 floats, valid while the index lives). Returns 0.
int cascade_router_params(void* p, int* out3, const float** step) {
    Index* ix = (Index*)p;
    out3[0] = ix->lv[ix->L - 1];
    out3[1] = ix->bgraph ? ix->bgraph->M : (ix->hnsw ? (int)ix->hnsw->M_ : 32);
    out3[2] = ix->bgraph ? ix->bgraph->efC : (ix->hnsw ? (int)ix->hnsw->ef_construction_ : 200);
    *step = ix->step.data();
    return 0;
}

// The image's graph section (frozen index): the compaction copies it before inserting the log's rows.
const void* cascade_base_graph(void* p) {
    Index* ix = (Index*)p;
    return ix->flat ? (const void*)ix->bgraph : nullptr;
}

// Bytes of private memory a frozen index holds beside the shared image (delta arrays, tombstones, the
// copied small arrays); the delta graph is the extension's and not counted. 0 for an owned index.
long long cascade_private_bytes(void* p) {
    Index* ix = (Index*)p;
    if (!ix->flat) return 0;
    long long b = (long long)ix->deleted.size() + 4LL * (ix->lo.size() + ix->step.size() + ix->proj.size() + ix->mean.size() + ix->cent.size());
    for (int l = 0; l < ix->L; ++l) b += 4LL * (ix->cal[l].size() + ix->err[l].size() + ix->tail[l].size());
    b += 4LL * (ix->perm.size() + ix->pos_of.size() + ix->cell_of.size());
    for (auto& v : ix->ov_codes) b += (long long)v.size();
    for (auto& v : ix->ov_pos) b += 4LL * v.size();
    return b;
}

// forward declarations of the scan API (defined below in this extern "C" block), used by the calibration
void* cascade_scan_begin(void* p, const float* q, int M, int P, int ef, const float* ch);
int cascade_scan_cell(void* scp, float tau2);
int cascade_scan_some(void* scp, float tau2, int ncells);
void cascade_scan_take(void* scp, int* ids, float* lb2);
void cascade_scan_end(void* scp);

// ===== E2: range-based build phases for the extension's parallel build (pgext/tp_build.c) =====
// Every entry point works on caller-provided arrays over a [lo, hi) range and touches nothing else, so
// PostgreSQL's maintenance workers call them over DSM arrays between barriers; the leader does the small
// serial steps (reductions, the PCA solve, centroid updates, sorting). No threads, no callbacks.

// m distinct indices out of n (sorted) from a seed; returns the count written (min(m, n))
int cascade_sample(uint64_t seed, int n, int m, int* out) {
    Rng rng((unsigned)(seed ^ (seed >> 32)));
    if (m >= n) { for (int i = 0; i < n; ++i) out[i] = i; return n; }
    std::vector<int> v = sample_without_replacement(n, m, rng);
    std::memcpy(out, v.data(), sizeof(int) * v.size());
    return (int)v.size();
}

// mean of the rows x[idx[0..m)] (leader)
void cascade_mean(const float* x, const int* idx, int m, int D, float* mean) {
    std::vector<double> mu(D, 0.0);
    for (int t = 0; t < m; ++t) { const float* r = x + (size_t)idx[t] * D; for (int k = 0; k < D; ++k) mu[k] += r[k]; }
    for (int k = 0; k < D; ++k) mean[k] = (float)(mu[k] / std::max(m, 1));
}

// upper triangle of sum_{t in [lo,hi)} (x[idx[t]] - mean)(x[idx[t]] - mean)^T added into cov (D x D doubles; the
// caller zeroes it), 256-row panels with 4-row register blocking as fit_head
void cascade_cov_partial(const float* x, const int* idx, int lo, int hi, int D, const float* mean, double* cov) {
    const int B = 256;
    std::vector<float> panel((size_t)D * B, 0.f);
    for (int s0 = lo; s0 < hi; s0 += B) {
        const int nb = std::min(B, hi - s0);
        for (int k = 0; k < D; ++k) std::fill(&panel[(size_t)k * B], &panel[(size_t)k * B] + B, 0.f);
        for (int b = 0; b < nb; ++b) {
            const float* r = x + (size_t)idx[s0 + b] * D;
            for (int k = 0; k < D; ++k) panel[(size_t)k * B + b] = r[k] - mean[k];
        }
        for (int i = 0; i < D; i += 4) {
            const int ni = std::min(4, D - i);
            const float* pi[4];
            for (int a = 0; a < ni; ++a) pi[a] = &panel[(size_t)(i + a) * B];
            for (int j = i; j < D; ++j) {
                const float* pj = &panel[(size_t)j * B];
                for (int a = 0; a < ni; ++a)
                    if (j >= i + a) cov[(size_t)(i + a) * D + j] += dot(pi[a], pj, B);
            }
        }
    }
}

// the PCA head from the summed covariance (upper triangle filled, m sampled rows): block subspace iteration
// with d3 + 64 vectors and a Rayleigh-Ritz step; proj = d3 x D orthonormal rows. Returns the variance share.
float cascade_pca_solve(double* cov, int m, int D, int d3, int iters, uint64_t seed, float* proj) {
    const double denom = std::max(m - 1, 1);
    double trace = 0;
    for (int i = 0; i < D; ++i) {
        for (int j = i; j < D; ++j) { double v = cov[(size_t)i * D + j] / denom; cov[(size_t)i * D + j] = v; cov[(size_t)j * D + i] = v; }
        trace += cov[(size_t)i * D + i];
    }
    Rng rng((unsigned)(seed ^ (seed >> 32)));
    const int r = std::min(D, d3 + 64);
    std::vector<double> Q((size_t)r * D), Y((size_t)r * D);
    for (auto& v : Q) v = rng.normal();
    orthonormalise(Q, D, r);
    auto mul = [&](const std::vector<double>& src, std::vector<double>& dst) {
        for (int j = 0; j < r; ++j) {
            const double* qj = &src[(size_t)j * D];
            double* yj = &dst[(size_t)j * D];
            for (int k = 0; k < D; ++k) yj[k] = ddot(&cov[(size_t)k * D], qj, D);
        }
    };
    for (int it = 0; it < iters; ++it) { mul(Q, Y); Q.swap(Y); orthonormalise(Q, D, r); }
    mul(Q, Y);
    std::vector<double> Bm((size_t)r * r), U, w;
    for (int a = 0; a < r; ++a)
        for (int b = a; b < r; ++b) { double v = ddot(&Q[(size_t)a * D], &Y[(size_t)b * D], D); Bm[(size_t)a * r + b] = v; Bm[(size_t)b * r + a] = v; }
    jacobi_eigen(Bm, r, U, w);
    std::vector<int> order(r);
    for (int i = 0; i < r; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return w[a] > w[b]; });
    double captured = 0;
    for (int t = 0; t < d3; ++t) {
        const int e = order[t];
        captured += std::max(w[e], 0.0);
        for (int k = 0; k < D; ++k) {
            double v = 0;
            for (int a = 0; a < r; ++a) v += Q[(size_t)a * D + k] * U[(size_t)a * r + e];
            proj[(size_t)t * D + k] = (float)v;
        }
    }
    return trace > 0 ? (float)(captured / trace) : 0.f;
}

// The PCA solve split for the parallel build (cascade_pca_solve does the same serially): prepare normalises
// the summed covariance (returns the trace), init draws Q (r x D, D contiguous per column, orthonormal), the
// participants compute Y[j] = cov Q[j] for columns [j_lo, j_hi), the elected one orthonormalises the swapped
// Q between iterations, finish does the Rayleigh-Ritz step and writes proj; returns the variance share.
double cascade_pca_prepare(double* cov, int m, int D) {
    const double denom = std::max(m - 1, 1);
    double trace = 0;
    for (int i = 0; i < D; ++i) {
        for (int j = i; j < D; ++j) { double v = cov[(size_t)i * D + j] / denom; cov[(size_t)i * D + j] = v; cov[(size_t)j * D + i] = v; }
        trace += cov[(size_t)i * D + i];
    }
    return trace;
}
int cascade_pca_rank(int D, int d3) { return std::min(D, d3 + 64); }
void cascade_pca_init(double* Q, int D, int r, uint64_t seed) {
    Rng rng((unsigned)(seed ^ (seed >> 32)));
    std::vector<double> q((size_t)r * D);
    for (auto& v : q) v = rng.normal();
    orthonormalise(q, D, r);
    std::memcpy(Q, q.data(), sizeof(double) * q.size());
}
void cascade_pca_mul_partial(const double* cov, int D, const double* Q, int j_lo, int j_hi, double* Y) {
    for (int j = j_lo; j < j_hi; ++j) {
        const double* qj = Q + (size_t)j * D;
        double* yj = Y + (size_t)j * D;
        for (int k = 0; k < D; ++k) yj[k] = ddot(cov + (size_t)k * D, qj, D);
    }
}
void cascade_orthonormalise(double* Q, int D, int r) {
    std::vector<double> q(Q, Q + (size_t)r * D);
    orthonormalise(q, D, r);
    std::memcpy(Q, q.data(), sizeof(double) * q.size());
}
float cascade_pca_finish(const double* Q, const double* Y, int D, int r, int d3, double trace, float* proj) {
    std::vector<double> Bm((size_t)r * r), U, w;
    for (int a = 0; a < r; ++a)
        for (int b = a; b < r; ++b) { double v = ddot(Q + (size_t)a * D, Y + (size_t)b * D, D); Bm[(size_t)a * r + b] = v; Bm[(size_t)b * r + a] = v; }
    jacobi_eigen(Bm, r, U, w);
    std::vector<int> order(r);
    for (int i = 0; i < r; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return w[a] > w[b]; });
    double captured = 0;
    for (int t = 0; t < d3; ++t) {
        const int e = order[t];
        captured += std::max(w[e], 0.0);
        for (int k = 0; k < D; ++k) {
            double v = 0;
            for (int a = 0; a < r; ++a) v += Q[(size_t)a * D + k] * U[(size_t)a * r + e];
            proj[(size_t)t * D + k] = (float)v;
        }
    }
    return trace > 0 ? (float)(captured / trace) : 0.f;
}

// H[i] = proj (x[i] - mean) for rows [lo, hi)
void cascade_project_rows(const float* x, int lo, int hi, int D, int d3, const float* proj, const float* mean, float* H) {
    std::vector<float> xc(D);
    for (int i = lo; i < hi; ++i) {
        const float* r = x + (size_t)i * D;
        for (int k = 0; k < D; ++k) xc[k] = r[k] - mean[k];
        for (int t = 0; t < d3; ++t) H[(size_t)i * d3 + t] = dot(&proj[(size_t)t * D], xc.data(), D);
    }
}

// per-dimension min / max of H over rows [lo, hi) (caller initialises mn = +inf, mx = -inf)
void cascade_minmax_partial(const float* H, int lo, int hi, int d3, float* mn, float* mx) {
    for (int i = lo; i < hi; ++i) {
        const float* h = H + (size_t)i * d3;
        for (int t = 0; t < d3; ++t) { mn[t] = std::min(mn[t], h[t]); mx[t] = std::max(mx[t], h[t]); }
    }
}

// signed int8 quantisation grid from the head's range (as cascade_build_vectors)
void cascade_lo_step(const float* mn, const float* mx, int d3, float* lo, float* step) {
    for (int t = 0; t < d3; ++t) {
        step[t] = mx[t] > mn[t] ? (mx[t] - mn[t]) / 255.f : 1.f;
        lo[t] = mn[t] + 128.f * step[t];
    }
}

// k-means init: C centroids from the training rows (train_idx[0..T)), seeded like kmeans_heads
void cascade_kmeans_init(const float* H, const int* train_idx, int T, int d3, int C, uint64_t seed, float* cent) {
    Rng rng((unsigned)(seed ^ (seed >> 32)) + 7u);
    std::vector<int> init = sample_without_replacement(T, std::min(C, T), rng);
    for (int c = 0; c < C; ++c) {
        const int src = c < (int)init.size() ? train_idx[init[c]] : train_idx[rng.below(T)];
        std::memcpy(cent + (size_t)c * d3, H + (size_t)src * d3, sizeof(float) * d3);
    }
}

// assign rows idx[lo..hi) (idx null: rows lo..hi themselves) to their nearest centroid: labels[t] (indexed by
// t, the range index), partial sums (C x d3 doubles) and counts (C) added into sum / cnt (the caller zeroes)
void cascade_kmeans_assign_partial(const float* H, const int* idx, int lo, int hi, int d3, const float* cent, int C,
                                   int* labels, double* sum, int* cnt) {
    std::vector<float> cn2(C);
    for (int c = 0; c < C; ++c) cn2[c] = dot(cent + (size_t)c * d3, cent + (size_t)c * d3, d3);
    for (int t = lo; t < hi; ++t) {
        const float* h = H + (size_t)(idx ? idx[t] : t) * d3;
        int best = 0; float bd = INFINITY;
        for (int c = 0; c < C; ++c) {
            float v = cn2[c] - 2.f * dot(h, cent + (size_t)c * d3, d3);
            if (v < bd) { bd = v; best = c; }
        }
        labels[t] = best;
        if (sum) { cnt[best]++; double* sc = &sum[(size_t)best * d3]; for (int k = 0; k < d3; ++k) sc[k] += h[k]; }
    }
}

// centroid update from the reduced sums; empty clusters are re-seeded from the training rows (seeded by iteration)
void cascade_kmeans_update(const float* H, const int* train_idx, int T, int d3, int C, const double* sum, const int* cnt,
                           uint64_t seed, int iter, float* cent) {
    Rng rng((unsigned)(seed ^ (seed >> 32)) + 13u * (unsigned)(iter + 1));
    for (int c = 0; c < C; ++c) {
        if (cnt[c] == 0) { std::memcpy(cent + (size_t)c * d3, H + (size_t)train_idx[rng.below(T)] * d3, sizeof(float) * d3); continue; }
        for (int k = 0; k < d3; ++k) cent[(size_t)c * d3 + k] = (float)(sum[(size_t)c * d3 + k] / cnt[c]);
    }
}

// one row's int8 codes (d3) and per-level error / tail norms from its vector x and head h
void cascade_encode_row(const float* x, const float* h, int D, int L, const int* lv, const float* lo, const float* step,
                        const float* mean, int8_t* codes, float* err_out, float* tail_out) {
    float xn2 = 0;
    for (int i = 0; i < D; ++i) { float t = x[i] - mean[i]; xn2 += t * t; }
    float hn2 = 0, e2 = 0;
    int i = 0;
    for (int l = 0; l < L; ++l) {
        for (; i < lv[l]; ++i) {
            float v = (h[i] - lo[i]) / step[i];
            int q = (int)std::lrint(v);
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            codes[i] = (int8_t)q;
            float d = h[i] - (lo[i] + (float)q * step[i]);
            e2 += d * d;
            hn2 += h[i] * h[i];
        }
        err_out[l] = std::sqrt(e2);
        tail_out[l] = std::sqrt(std::max(xn2 - hn2, 0.f));
    }
}

// Offsets of a flat image assembled in place by the extension: fills h (magic, sizes, offsets, total_bytes)
// for n positions in C cells with the given per-cell counts (cell_cnt, C of them), npos_of id slots, cal_cap
// slots per calibration level and a graph blob of graph_bytes. Returns total_bytes. The caller fills the arrays
// (perm, pos_of, cell_of, deleted, err/tail, blk offsets + codes + positions, graph) and the small vectors.
long long cascade_flat_layout(void* hdr, int n, int D, int C, int L, const int* lv, int M, int efC, int npos_of,
                              const uint64_t* cal_cap, const int* cell_cnt, uint64_t graph_bytes) {
    FlatHeader* h = (FlatHeader*)hdr;
    std::memset(h, 0, sizeof(FlatHeader));
    const int d3 = lv[L - 1];
    FlatLayout lay;
    lay.reserve(sizeof(FlatHeader));
    h->magic = TPFL_MAGIC; h->version = TPFL_VERSION;
    h->n = n; h->D = D; h->C = C; h->L = L;
    for (int l = 0; l < L; ++l) { h->lv[l] = lv[l]; h->ch[l] = 1.f; }
    h->M = M; h->efC = efC; h->npos_of = npos_of;
    h->lo = lay.reserve((uint64_t)d3 * 4); h->step = lay.reserve((uint64_t)d3 * 4);
    h->proj = lay.reserve((uint64_t)d3 * D * 4); h->mean = lay.reserve((uint64_t)D * 4);
    h->cent = lay.reserve((uint64_t)C * d3 * 4);
    for (int l = 0; l < L; ++l) { h->cal_n[l] = 0; h->cal_cap[l] = cal_cap[l]; h->cal[l] = lay.reserve(cal_cap[l] * 4); }
    h->perm = lay.reserve((uint64_t)n * 4); h->pos_of = lay.reserve((uint64_t)npos_of * 4);
    h->cell_of = lay.reserve((uint64_t)n * 4); h->deleted = lay.reserve((uint64_t)n);
    for (int l = 0; l < L; ++l) { h->err[l] = lay.reserve((uint64_t)n * 4); h->tail[l] = lay.reserve((uint64_t)n * 4); }
    h->blk_code_off = lay.reserve((uint64_t)(C + 1) * 8); h->blk_pos_off = lay.reserve((uint64_t)(C + 1) * 8);
    uint64_t codes_total = 0, pos_total = 0;
    for (int c = 0; c < C; ++c) { codes_total += (uint64_t)cell_cnt[c] * d3; pos_total += (uint64_t)cell_cnt[c]; }
    h->blk_codes = lay.reserve(codes_total); h->blk_pos = lay.reserve(pos_total * 4);
    h->graph_bytes = graph_bytes; h->graph = lay.reserve(graph_bytes);
    h->total_bytes = lay.reserve(0);
    return (long long)h->total_bytes;
}

// Calibration of one query (corpus row t; x id-major over every row, H its head): the exact top-(CAL_K+1) through
// the frozen index's own scan (certificate mode) with exact distances from x, then per level the critical cosine
// g* of every (t, neighbour) pair (calibrate_index's formula). out: L * CAL_K floats (level-major), cnt[l] written per
// level. Returns 1 when the query produced CAL_K neighbours, 0 otherwise.
// order: id -> row of x / H (null = identity; the parallel build numbers ids in TID order while its rows sit in scan order)
int cascade_calibrate_query(void* frozen, const float* x, int D, const float* H, const int* order, int t, float* out, int* cnt) {
    Index* ix = (Index*)frozen;
    const int L = ix->L, d3 = ix->lv[L - 1];
    for (int l = 0; l < L; ++l) cnt[l] = 0;
    auto row = [&](int id) { return (size_t)(order ? order[id] : id); };
    const float* q = x + row(t) * D;
    const float ones[4] = {1, 1, 1, 1};
    void* sc = cascade_scan_begin(ix, q, 100, ix->C, 100, ones);
    if (!sc) return 0;
    const float slack = 1.f + 1e-5f;
    std::vector<std::pair<float, int>> top;  // (d2, id), kept sorted, size <= CAL_K + 1 (t itself excluded)
    float tau2 = INFINITY;
    std::vector<int> ids; std::vector<float> lb;
    for (;;) {
        int m = cascade_scan_cell(sc, tau2);
        if (m < 0) break;
        if (m == 0) continue;
        ids.resize(m); lb.resize(m);
        cascade_scan_take(sc, ids.data(), lb.data());
        for (int i = 0; i < m; ++i) {
            if (lb[i] > tau2 * slack) break;
            const int id = ids[i];
            if (id == t || id < 0) continue;
            const float d2 = sq_dist_full(q, x + row(id) * D, D);
            if ((int)top.size() == CAL_K && d2 >= top.back().first) continue;
            top.push_back({d2, id});
            std::sort(top.begin(), top.end());
            if ((int)top.size() > CAL_K) top.pop_back();
            if ((int)top.size() == CAL_K) tau2 = top.back().first;
        }
    }
    cascade_scan_end(sc);
    if ((int)top.size() < CAL_K) return 0;
    const float tau2f = top.back().first;
    const int pos_t = ix->pos_of_id(t);
    if (pos_t < 0) return 0;
    std::vector<float> qa(d3);
    const float* ht = H + row(t) * d3;
    for (int i = 0; i < d3; ++i) qa[i] = ht[i] - ix->lo[i];
    for (int l = 0; l < L; ++l) {
        const int hl = ix->lv[l];
        const double a = ix->tail_at(l, pos_t);
        for (auto& pr : top) {
            const int pos_r = ix->pos_of_id(pr.second);
            if (pos_r < 0) continue;
            const double hd = sq_dist_codes(qa.data(), ix->step.data(), fg_code(ix->bgraph, pos_r), hl);
            double hlb = std::sqrt(std::max(hd, 0.0)) - ix->err_at(l, pos_r);
            if (hlb < 0) hlb = 0;
            const double b = ix->tail_at(l, pos_r);
            const double num = hlb * hlb + a * a + b * b - (double)tau2f, den = 2 * a * b;
            double g = den > 0 ? num / den : (num > 0 ? 2.0 : -2.0);
            out[(size_t)l * CAL_K + cnt[l]++] = (float)std::max(-2.0, std::min(2.0, g));
        }
    }
    return 1;
}

// The router's hits for one query (parity gate / routing experiments): the top-M positions of the beam
// (ef, base + delta graphs) as original ids with their scores (||key - c||_w^2 up to a per-query constant).
// Returns the count written (<= M).
int cascade_route_hits(void* p, const float* q, int M, int ef, int* ids_out, float* scores_out) {
    Index* ix = (Index*)p;
    Scratch s;
    long long st[3] = {0, 0, 0};
    Prof pf;
    route_query(ix, s, q, M, ix->C, std::max(ef, M), st, pf);
    int k = 0;
    for (auto& pr : s.res) {
        if (k >= M) break;
        if (!ix->reduced() && ix->deleted[pr.second]) continue;
        ids_out[k] = ix->perm_at(pr.second); scores_out[k] = pr.first; ++k;
    }
    return k;
}

// c_h at miss probability delta from one level's sorted cosines
float cascade_ch_from_sorted(const float* g, int n, double delta) {
    std::vector<float> v(g, g + n);
    return ch_for_delta(v, delta);
}

// ---- incremental maintenance for the extension: the inserting backend encodes once and writes the
// record (cell, codes, err, tail) to the index's log; every backend applies records to its own copy.
// Returns the cell the point belongs to (nearest level-3 head centroid, as cascade_insert).
int cascade_encode(void* p, const float* x, int8_t* codes_out, float* err_out, float* tail_out) {
    Index* ix = (Index*)p;
    const int D = ix->D, L = ix->L, d3 = ix->lv[L - 1], C = ix->C;
    std::vector<float> qc(D), h(d3);
    for (int k = 0; k < D; ++k) qc[k] = x[k] - ix->mean[k];
    for (int r = 0; r < d3; ++r) h[r] = dot(&ix->proj[(size_t)r * D], qc.data(), D);
    int best = 0; float bd = INFINITY;
    for (int c = 0; c < C; ++c) {
        float d = sq_dist_full(h.data(), &ix->cent[(size_t)c * d3], d3);
        if (d < bd) { bd = d; best = c; }
    }
    encode_point(ix, x, h.data(), codes_out, err_out, tail_out);
    return best;
}

// Apply one encoded insert: append to the cell's overflow block, add to the router (pos = id's
// position). ids must be new (>= total()) or previously unused; the vectors are not stored.
int cascade_insert_codes(void* p, int id, int cell, const int8_t* codes, const float* err, const float* tail) {
    Index* ix = (Index*)p;
    const int L = ix->L, d3 = ix->lv[L - 1];
    if (cell < 0 || cell >= ix->C || id < 0) return -1;
    if (ix->pos_of_id(id) >= 0) return -2;  // id in use
    if (id < ix->fb_npos_of) return -3;      // frozen: an unused id inside the base's id range cannot be mapped (the extension never reuses ids)
    const int pos = ix->total();
    {
        const int did = id - ix->fb_npos_of;  // delta-relative id (== id for an owned index)
        if (did >= (int)ix->pos_of.size()) ix->pos_of.resize((size_t)did + 1, -1);
        ix->pos_of[did] = pos;
    }
    ix->perm.push_back(id);
    ix->cell_of.push_back(cell);
    ix->deleted.push_back(0);
    if (ix->flat) ix->n_delta_live++;
    for (int l = 0; l < L; ++l) { ix->err[l].push_back(err[l]); ix->tail[l].push_back(tail[l]); }
    if (!ix->xfull.empty()) ix->xfull.resize(ix->xfull.size() + (size_t)ix->D, 0.f);  // keep vec() addressable (never read for a sketch-only index)
    ix->ov_codes[cell].insert(ix->ov_codes[cell].end(), codes, codes + d3);
    ix->ov_pos[cell].push_back(pos);
    if (!ix->flat && ix->hnsw && !ix->reduced()) {
        ix->space->prm.query_float = false;
        if (ix->hnsw->cur_element_count >= ix->hnsw->max_elements_) ix->hnsw->resizeIndex(std::max<size_t>(ix->hnsw->max_elements_ * 2, 1024));
        ix->hnsw->addPoint(codes, (size_t)pos);
        ix->space->prm.query_float = true;
        ix->rcodes.insert(ix->rcodes.end(), codes, codes + d3);
        ix->rg.dirty = true;
    }
    // frozen: the router of the appended positions is the extension's delta graph (cascade_set_delta_graph)
    ix->n_inserted++;
    return pos;
}

// out = {n0, total, D, C, L, lv0, lv1, lv2, lv3, n_inserted, n_deleted, n_compacted, calibrated(0/1), router_nodes}
int cascade_info(void* p, long long* out) {
    Index* ix = (Index*)p;
    out[0] = ix->n0; out[1] = ix->total(); out[2] = ix->D; out[3] = ix->C; out[4] = ix->L;
    for (int l = 0; l < 4; ++l) out[5 + l] = ix->lv[l];
    out[9] = ix->n_inserted; out[10] = ix->n_deleted; out[11] = ix->n_compacted;
    out[12] = ix->cal[0].empty() ? 0 : 1;
    out[13] = ix->hnsw ? (long long)ix->hnsw->cur_element_count : (ix->bgraph ? (long long)ix->bgraph->n + (ix->dgraph ? ix->dgraph->n : 0) : 0);
    return 14;
}

float cascade_var_captured(void* p) { return ((Index*)p)->var_captured; }

// ---- two-phase scan: route once, then one cell per call; the caller computes exact distances ----
// (the PostgreSQL extension fetches the survivors' heap rows between calls, so no callback into the
// server runs inside a C++ frame). ch: per-level tail cosines, null = ix->ch.
struct ScanState {
    Index* ix = nullptr;
    Scratch s;
    std::vector<std::pair<float, int>> merged;  // cascade_scan_rest: survivors of several cells
    std::vector<float> q;
    float ch[4] = {1, 1, 1, 1};
    int ci = 0, nP = 0;
    int P_asked = 0;          // the P of cascade_scan_begin: the rest of the cells is ranked once P exceeds the hit count
    long long hits = 0, cells_hit = 0;
    Prof pf;
};

void* cascade_scan_begin(void* p, const float* q, int M, int P, int ef, const float* ch) {
    Index* ix = (Index*)p;
    ScanState* sc = new ScanState();
    sc->ix = ix;
    sc->q.assign(q, q + ix->D);
    for (int l = 0; l < ix->L; ++l) sc->ch[l] = ch ? ch[l] : ix->ch[l];
    long long st[3] = {0, 0, 0};
    sc->s.ops = 0;
    route_query(ix, sc->s, sc->q.data(), M, P, ef, st, sc->pf);
    if (sc->s.corrupt[0]) { std::string m = sc->s.corrupt; delete sc; throw std::runtime_error(m); }
    sc->hits = st[1]; sc->cells_hit = st[2];
    sc->P_asked = P;
    sc->nP = std::min<int>(P, (int)sc->s.cells.size());
    return sc;
}

// Iterative scan: rewind the cell cursor so the cells already scanned are cascaded again
// under the caller's new, larger tau2, and let the scan continue over the first P cells of the router's order
// (the hit cells first, then every other non-empty cell by centroid distance, ranked here the first time P
// exceeds the hit count). The route is not repeated. Returns the number of cells the scan will cover.
int cascade_scan_restart(void* scp, int P) {
    ScanState* sc = (ScanState*)scp;
    if (P > sc->P_asked) {
        if (sc->P_asked <= (int)sc->cells_hit) append_rest_cells(sc->ix, sc->s);  // not yet ranked by cascade_scan_begin
        sc->P_asked = P;
    }
    sc->nP = std::min<int>(P, (int)sc->s.cells.size());
    sc->ci = 0;
    return sc->nP;
}

// Cascade over the next cell under tau2 (the caller's current k-th exact distance, INFINITY at
// first). Returns the number of survivors (take them with cascade_scan_take), or -1 when every
// one of the P cells has been scanned.
int cascade_scan_cell(void* scp, float tau2) {
    ScanState* sc = (ScanState*)scp;
    if (sc->ci >= sc->nP) return -1;
    const int c = sc->s.cells[sc->ci++].second;
    scan_cell(sc->ix, sc->s, c, tau2, sc->ch, sc->pf);
    if (sc->s.corrupt[0]) throw std::runtime_error(sc->s.corrupt);
    return (int)sc->s.cand.size();
}

// Cascade over every remaining cell under tau2 and merge their survivors into one list sorted by
// bound (take it with cascade_scan_take). Fetching in global bound order and stopping at the first
// bound above tau fetches the minimum any bound-ordered exact phase must; the price is that these
// cells are pruned with the tau the caller has now, not with the tighter one cell-by-cell would give.
// Returns the number of survivors, or -1 when nothing is left.
int cascade_scan_rest(void* scp, float tau2) { return cascade_scan_some(scp, tau2, INT_MAX); }

// The next `ncells` cells (at most) under tau2, merged by bound as cascade_scan_rest: an iterative scan's later
// batches take the rest in growing chunks, so that the tau their fetches tighten prunes the chunks after them.
int cascade_scan_some(void* scp, float tau2, int ncells) {
    ScanState* sc = (ScanState*)scp;
    if (sc->ci >= sc->nP) return -1;
    const int stop = (ncells >= sc->nP - sc->ci) ? sc->nP : sc->ci + std::max(ncells, 1);
    sc->merged.clear();
    while (sc->ci < stop) {
        const int c = sc->s.cells[sc->ci++].second;
        scan_cell(sc->ix, sc->s, c, tau2, sc->ch, sc->pf);
        if (sc->s.corrupt[0]) throw std::runtime_error(sc->s.corrupt);
        sc->merged.insert(sc->merged.end(), sc->s.cand.begin(), sc->s.cand.end());
    }
    std::sort(sc->merged.begin(), sc->merged.end());
    sc->s.cand.swap(sc->merged);
    return (int)sc->s.cand.size();
}

// Survivors of the last cell (or of cascade_scan_rest): original ids (the caller maps them to rows)
// and their squared lower bounds, in bound order. A survivor whose bound exceeds tau2 can be skipped.
void cascade_scan_take(void* scp, int* ids, float* lb2) {
    ScanState* sc = (ScanState*)scp;
    for (size_t i = 0; i < sc->s.cand.size(); ++i) { ids[i] = sc->ix->perm_at(sc->s.cand[i].second); lb2[i] = sc->s.cand[i].first; }
}

// out = {dimension-ops so far, router hits asked, cells with a hit, cells scanned so far, cells ranked}
void cascade_scan_stats(void* scp, long long* out) {
    ScanState* sc = (ScanState*)scp;
    out[0] = sc->s.ops; out[1] = sc->hits; out[2] = sc->cells_hit; out[3] = sc->ci; out[4] = sc->nP;
}

void cascade_scan_end(void* scp) { delete (ScanState*)scp; }

}  // extern "C"
