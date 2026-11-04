/* Flat HNSW router graph over int8 codes — the one layout shared by the extension's C builder
 * (pgext/tp_graph.c: builds it in a DSM area with per-node LWLocks, or in private memory for a
 * compaction / a backend's delta) and the core's beam search (csrc/cascade.cpp: reads it in place,
 * as the router section of a flat index image). Plain C, no PostgreSQL types: position-independent,
 * every array at a byte offset from the header, 64-byte aligned.
 *
 *   codes    int8  [n_cap][d3]        the node's int8 head (its router code)
 *   sc       float [n_cap]            sum_i w2[i] * code[i]^2 (the beam's per-node constant)
 *   w2       float [d3]               step[i]^2: the weights of the int8 distance
 *   level    int32 [n_cap]            the node's top level (-1: not inserted yet)
 *   links0   int32 [n_cap][M0 + 1]    level-0 list: count, then ids (pos space)
 *   upidx    int32 [n_cap]            first upper block of the node (its level-1 list), -1 if level 0
 *   uplinks  int32 [up_cap][M + 1]    upper lists: block k = (count, ids); node's level l list = block upidx + l - 1
 * Node ids are positions in the index they belong to (base graph: 0..n-1; a backend's delta graph:
 * 0..k-1 standing for positions nb..nb+k-1). A reader never needs more than the header to walk it.
 */
#ifndef TRIGPOINT_FLAT_GRAPH_H
#define TRIGPOINT_FLAT_GRAPH_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#define FG_MAGIC 0x47464c46u	/* "FLFG" */
#define FG_VERSION 1u

typedef struct FlatGraph
{
	uint32_t	magic;
	uint32_t	version;
	int32_t		n_cap;			/* nodes the area can hold */
	int32_t		n;				/* nodes inserted so far (highest inserted id + 1 for the base; the builder sets it) */
	int32_t		d3;
	int32_t		M;				/* upper-level degree (and the selection degree at every level) */
	int32_t		M0;				/* level-0 degree (2 M) */
	int32_t		efC;
	int32_t		maxlevel;
	int32_t		enterpoint;		/* -1 while empty */
	int32_t		up_cap;			/* upper blocks available */
	int32_t		up_used;		/* upper blocks handed out (bump allocator; the builder updates it) */
	uint64_t	seed;			/* level draws: level(pos) = floor(-ln(u(seed, pos)) * mult) */
	double		mult;			/* 1 / ln(M) */
	uint64_t	codes_off;
	uint64_t	sc_off;
	uint64_t	w2_off;
	uint64_t	level_off;
	uint64_t	links0_off;
	uint64_t	upidx_off;
	uint64_t	uplinks_off;
	uint64_t	total_bytes;
} FlatGraph;

#define FG_ALIGN(x) (((uint64_t) (x) + 63) & ~(uint64_t) 63)

/* upper blocks to reserve for n_cap nodes: the expected number is n / (M - 1); keep 3x plus slack */
static inline int32_t
fg_up_cap(int32_t n_cap, int32_t M)
{
	int64_t		c = (int64_t) n_cap * 3 / (M > 1 ? M - 1 : 1) + 1024;

	return c > 0x7fffffff ? 0x7fffffff : (int32_t) c;
}

/* fills the offsets of h for the given capacities; returns the bytes the whole area needs */
static inline uint64_t
fg_layout(FlatGraph *h, int32_t n_cap, int32_t d3, int32_t M, int32_t up_cap)
{
	uint64_t	off = FG_ALIGN(sizeof(FlatGraph));

	h->magic = FG_MAGIC;
	h->version = FG_VERSION;
	h->n_cap = n_cap;
	h->d3 = d3;
	h->M = M;
	h->M0 = 2 * M;
	h->up_cap = up_cap;
	h->codes_off = off;
	off = FG_ALIGN(off + (uint64_t) n_cap * (uint64_t) d3);
	h->sc_off = off;
	off = FG_ALIGN(off + (uint64_t) n_cap * 4);
	h->w2_off = off;
	off = FG_ALIGN(off + (uint64_t) d3 * 4);
	h->level_off = off;
	off = FG_ALIGN(off + (uint64_t) n_cap * 4);
	h->links0_off = off;
	off = FG_ALIGN(off + (uint64_t) n_cap * (uint64_t) (h->M0 + 1) * 4);
	h->upidx_off = off;
	off = FG_ALIGN(off + (uint64_t) n_cap * 4);
	h->uplinks_off = off;
	off = FG_ALIGN(off + (uint64_t) up_cap * (uint64_t) (M + 1) * 4);
	h->total_bytes = off;
	return off;
}

static inline int8_t *fg_codes(FlatGraph *h) { return (int8_t *) ((char *) h + h->codes_off); }
static inline const int8_t *fg_code(const FlatGraph *h, int32_t i) { return (const int8_t *) ((const char *) h + h->codes_off) + (size_t) i * (size_t) h->d3; }
static inline float *fg_sc(FlatGraph *h) { return (float *) ((char *) h + h->sc_off); }
static inline const float *fg_sc_c(const FlatGraph *h) { return (const float *) ((const char *) h + h->sc_off); }
static inline float *fg_w2(FlatGraph *h) { return (float *) ((char *) h + h->w2_off); }
static inline const float *fg_w2_c(const FlatGraph *h) { return (const float *) ((const char *) h + h->w2_off); }
static inline int32_t *fg_level(FlatGraph *h) { return (int32_t *) ((char *) h + h->level_off); }
static inline const int32_t *fg_level_c(const FlatGraph *h) { return (const int32_t *) ((const char *) h + h->level_off); }
static inline int32_t *fg_links0(FlatGraph *h, int32_t i) { return (int32_t *) ((char *) h + h->links0_off) + (size_t) i * (size_t) (h->M0 + 1); }
static inline const int32_t *fg_links0_c(const FlatGraph *h, int32_t i) { return (const int32_t *) ((const char *) h + h->links0_off) + (size_t) i * (size_t) (h->M0 + 1); }
static inline int32_t *fg_upidx(FlatGraph *h) { return (int32_t *) ((char *) h + h->upidx_off); }
static inline const int32_t *fg_upidx_c(const FlatGraph *h) { return (const int32_t *) ((const char *) h + h->upidx_off); }
/* the list of node i at level l >= 1 (NULL when the node has no such level) */
static inline int32_t *fg_links(FlatGraph *h, int32_t i, int32_t l)
{
	int32_t		k = fg_upidx(h)[i];

	if (l < 1 || k < 0 || l > fg_level(h)[i] || (int64_t) k + (l - 1) >= h->up_cap)
		return NULL;
	return (int32_t *) ((char *) h + h->uplinks_off) + ((size_t) k + (size_t) (l - 1)) * (size_t) (h->M + 1);
}
static inline const int32_t *fg_links_c(const FlatGraph *h, int32_t i, int32_t l)
{
	int32_t		k = fg_upidx_c(h)[i];

	if (l < 1 || k < 0 || l > fg_level_c(h)[i] || (int64_t) k + (l - 1) >= h->up_cap)	/* a corrupt upidx stays inside the area */
		return NULL;
	return (const int32_t *) ((const char *) h + h->uplinks_off) + ((size_t) k + (size_t) (l - 1)) * (size_t) (h->M + 1);
}

/* splitmix64 on (seed, pos): the level draw, identical in every process and every build */
static inline uint64_t
fg_mix(uint64_t z)
{
	z += 0x9e3779b97f4a7c15ull;
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
	z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
	return z ^ (z >> 31);
}

static inline int32_t
fg_draw_level(uint64_t seed, int32_t pos, double mult)
{
	uint64_t	r = fg_mix(seed ^ ((uint64_t) (uint32_t) pos << 1 | 1ull) ^ (seed << 32));
	double		u = ((double) (r >> 11) + 0.5) / 9007199254740992.0;	/* (0, 1) */
	double		lv = -log(u) * mult;

	return lv > 64 ? 64 : (int32_t) lv;
}

#endif
