/* The extension's HNSW builder over the flat graph layout (csrc/flat_graph.h) — E2.
 *
 * One implementation for three uses: the parallel CREATE INDEX (the graph lives in the build's DSM
 * area, every participant inserts its positions, per-node LWLocks in a separate shared array), the
 * compaction (a private copy of the generation's graph, the log's rows inserted single-process) and a
 * backend's delta (a small private graph over the positions appended since the image, grown as the
 * log is replayed). Insertion follows hnswlib's addPoint: deterministic level per position (seed),
 * greedy descent through the upper layers, an ef_construction beam per level, hnswlib's neighbour
 * heuristic for selection and for pruning a full list, bidirectional links. Lock discipline follows
 * pgvector's: never two node locks at once (a list is copied out under a shared lock, rewritten under
 * an exclusive one), so concurrent inserters cannot deadlock. The int8 weighted distance is the core's.
 */
#ifndef TP_GRAPH_H
#define TP_GRAPH_H

#include "postgres.h"

#include "port/atomics.h"
#include "storage/lwlock.h"

#include "flat_graph.h"

typedef struct TpGraphShared		/* the lock array beside a DSM graph (not part of the graph image) */
{
	LWLock		entry_lock;
	pg_atomic_uint32 up_alloc;		/* bump allocator of upper blocks */
	int			n_cap;
	LWLock		locks[FLEXIBLE_ARRAY_MEMBER];
} TpGraphShared;

typedef struct TpGraphCtx
{
	FlatGraph  *g;
	TpGraphShared *sh;				/* NULL: private, single-process graph */
} TpGraphCtx;

typedef struct TpCand
{
	float		d;
	int32		id;
} TpCand;

typedef struct TpGraphScratch
{
	uint32	   *visited;			/* epoch stamps, n_cap */
	uint32		epoch;
	int32		n_cap;
	TpCand	   *cand;				/* min-heap of candidates to expand */
	TpCand	   *res;				/* max-heap of the ef best */
	TpCand	   *tmp;				/* heuristic work */
	int32	   *list;				/* copy of one neighbour list */
	int			cap;				/* heap capacity (ef_construction + M0 + 2) */
} TpGraphScratch;

extern Size tp_graph_bytes(int n_cap, int d3, int M);
extern Size tp_graph_shared_bytes(int n_cap);
extern void tp_graph_init(FlatGraph *g, Size bytes, int n_cap, int d3, int M, int efC, const float *step, uint64 seed);
extern void tp_graph_shared_init(TpGraphShared *sh, int n_cap, int tranche_id);
extern void tp_graph_set_code(FlatGraph *g, int pos, const int8 *code);
extern TpGraphScratch *tp_graph_scratch_new(const FlatGraph *g);
extern void tp_graph_scratch_free(TpGraphScratch *sc);
extern void tp_graph_insert(TpGraphCtx *ctx, int pos, TpGraphScratch *sc);
/* private graphs (malloc'd): a new empty one, and a copy with a larger capacity (the old one is freed) */
extern FlatGraph *tp_graph_private_new(int n_cap, int d3, int M, int efC, const float *step, uint64 seed);
extern FlatGraph *tp_graph_private_grow(FlatGraph *g, int new_cap);
extern FlatGraph *tp_graph_private_copy(const FlatGraph *src, int new_cap);
extern FlatGraph *tp_graph_compact(const FlatGraph *src, const int32 *newpos, int n_new, int new_cap);	/* H13: the live nodes only, renumbered, lists repaired */	/* a private copy of any graph (e.g. a base graph) with room for new_cap nodes */

#endif
