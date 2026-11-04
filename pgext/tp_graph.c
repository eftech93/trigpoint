/* HNSW construction over the flat graph layout — see tp_graph.h. */
#include "tp_graph.h"

#include <math.h>
#include <string.h>

#include "miscadmin.h"
#include "trigpoint_core.h"

/* ------------------------------------------------------------------------------------------------ */
/* layout */

Size
tp_graph_bytes(int n_cap, int d3, int M)
{
	FlatGraph	h;

	return (Size) fg_layout(&h, n_cap, d3, M, fg_up_cap(n_cap, M));
}

Size
tp_graph_shared_bytes(int n_cap)
{
	return offsetof(TpGraphShared, locks) + sizeof(LWLock) * (Size) n_cap;
}

void
tp_graph_init(FlatGraph *g, Size bytes, int n_cap, int d3, int M, int efC, const float *step, uint64 seed)
{
	FlatGraph	h;
	uint64		need = fg_layout(&h, n_cap, d3, M, fg_up_cap(n_cap, M));
	int32	   *lv,
			   *ui;
	float	   *w2;
	int			i;

	if (need > bytes)
		elog(ERROR, "trigpoint: graph area too small (%zu < " UINT64_FORMAT " bytes)", bytes, need);
	memset(g, 0, need);
	*g = h;
	g->n = 0;
	g->efC = efC;
	g->maxlevel = -1;
	g->enterpoint = -1;
	g->up_used = 0;
	g->seed = seed;
	g->mult = 1.0 / log((double) Max(M, 2));
	lv = fg_level(g);
	ui = fg_upidx(g);
	for (i = 0; i < n_cap; i++)
	{
		lv[i] = -1;
		ui[i] = -1;
	}
	w2 = fg_w2(g);
	for (i = 0; i < d3; i++)
		w2[i] = step[i] * step[i];
}

void
tp_graph_shared_init(TpGraphShared *sh, int n_cap, int tranche_id)
{
	int			i;

	LWLockInitialize(&sh->entry_lock, tranche_id);
	pg_atomic_init_u32(&sh->up_alloc, 0);
	sh->n_cap = n_cap;
	for (i = 0; i < n_cap; i++)
		LWLockInitialize(&sh->locks[i], tranche_id);
}

void
tp_graph_set_code(FlatGraph *g, int pos, const int8 *code)
{
	int8	   *dst = fg_codes(g) + (size_t) pos * (size_t) g->d3;
	const float *w2 = fg_w2_c(g);
	float		acc = 0;
	int			i;

	memcpy(dst, code, (size_t) g->d3);
	for (i = 0; i < g->d3; i++)
		acc += w2[i] * (float) code[i] * (float) code[i];
	fg_sc(g)[pos] = acc;
}

/* ------------------------------------------------------------------------------------------------ */
/* private (malloc'd) graphs */

FlatGraph *
tp_graph_private_new(int n_cap, int d3, int M, int efC, const float *step, uint64 seed)
{
	Size		bytes = tp_graph_bytes(n_cap, d3, M);
	FlatGraph  *g = (FlatGraph *) malloc(bytes);

	if (g == NULL)
		return NULL;
	tp_graph_init(g, bytes, n_cap, d3, M, efC, step, seed);
	return g;
}

/* copy src into a fresh private graph with room for new_cap nodes (>= src->n_cap) */
FlatGraph *
tp_graph_private_copy(const FlatGraph *src, int new_cap)
{
	FlatGraph  *g;
	Size		bytes;

	if (new_cap < src->n_cap)
		new_cap = src->n_cap;
	bytes = tp_graph_bytes(new_cap, src->d3, src->M);
	g = (FlatGraph *) malloc(bytes);
	if (g == NULL)
		return NULL;
	tp_graph_init(g, bytes, new_cap, src->d3, src->M, src->efC, fg_w2_c(src), src->seed);
	/* tp_graph_init squares what it is given as the steps; the source holds w2 = step² already, so copy the
	 * weights verbatim (otherwise every copy carries step⁴ against node norms computed with step², and the
	 * beam ranks by norm: compaction recall 1.0 → 0.21 on a 12k table, grown delta graphs mis-scaled). */
	memcpy(fg_w2(g), fg_w2_c(src), sizeof(float) * (size_t) src->d3);
	/* the upper-block capacity grows with n_cap; copy block by block through the node index */
	memcpy(fg_codes(g), fg_code(src, 0), (size_t) src->n_cap * (size_t) src->d3);
	memcpy(fg_sc(g), fg_sc_c(src), sizeof(float) * (size_t) src->n_cap);
	memcpy(fg_level(g), fg_level_c(src), sizeof(int32) * (size_t) src->n_cap);
	memcpy(fg_links0(g, 0), fg_links0_c(src, 0), sizeof(int32) * (size_t) src->n_cap * (size_t) (src->M0 + 1));
	memcpy(fg_upidx(g), fg_upidx_c(src), sizeof(int32) * (size_t) src->n_cap);
	if (src->up_used > g->up_cap)
		elog(ERROR, "trigpoint: graph copy has fewer upper blocks than the source uses");
	memcpy((char *) g + g->uplinks_off, (const char *) src + src->uplinks_off, sizeof(int32) * (size_t) src->up_used * (size_t) (src->M + 1));
	g->up_used = src->up_used;
	g->n = src->n;
	g->maxlevel = src->maxlevel;
	g->enterpoint = src->enterpoint;
	return g;
}

FlatGraph *
tp_graph_private_grow(FlatGraph *g, int new_cap)
{
	FlatGraph  *ng = tp_graph_private_copy(g, new_cap);

	if (ng != NULL)
		free(g);
	return ng;
}

/* ------------------------------------------------------------------------------------------------ */
/* scratch */

TpGraphScratch *
tp_graph_scratch_new(const FlatGraph *g)
{
	TpGraphScratch *sc = (TpGraphScratch *) palloc0(sizeof(TpGraphScratch));

	sc->n_cap = g->n_cap;
	sc->visited = (uint32 *) palloc0(sizeof(uint32) * (Size) g->n_cap);
	sc->epoch = 0;
	sc->cap = g->efC + g->M0 + 2;
	sc->cand = (TpCand *) palloc(sizeof(TpCand) * (Size) (sc->cap * 4 + 16));
	sc->res = (TpCand *) palloc(sizeof(TpCand) * (Size) (sc->cap + 2));
	sc->tmp = (TpCand *) palloc(sizeof(TpCand) * (Size) (sc->cap + 2));
	sc->list = (int32 *) palloc(sizeof(int32) * (Size) (g->M0 + 2));
	return sc;
}

void
tp_graph_scratch_free(TpGraphScratch *sc)
{
	if (sc == NULL)
		return;
	pfree(sc->visited);
	pfree(sc->cand);
	pfree(sc->res);
	pfree(sc->tmp);
	pfree(sc->list);
	pfree(sc);
}

static void
scratch_fit(TpGraphScratch *sc, const FlatGraph *g)
{
	if (g->n_cap > sc->n_cap)
	{
		sc->visited = (uint32 *) repalloc(sc->visited, sizeof(uint32) * (Size) g->n_cap);
		memset(sc->visited + sc->n_cap, 0, sizeof(uint32) * (Size) (g->n_cap - sc->n_cap));
		sc->n_cap = g->n_cap;
	}
}

/* ------------------------------------------------------------------------------------------------ */
/* heaps over TpCand (min-heap on d for candidates, max-heap for results) */

static inline void
heap_push(TpCand *h, int *n, TpCand v, bool minheap)
{
	int			i = (*n)++;

	h[i] = v;
	while (i > 0)
	{
		int			p = (i - 1) / 2;
		bool		up = minheap ? (h[i].d < h[p].d) : (h[i].d > h[p].d);

		if (!up)
			break;
		{
			TpCand		t = h[i];

			h[i] = h[p];
			h[p] = t;
		}
		i = p;
	}
}

static inline TpCand
heap_pop(TpCand *h, int *n, bool minheap)
{
	TpCand		top = h[0];
	int			i = 0;

	h[0] = h[--(*n)];
	for (;;)
	{
		int			l = 2 * i + 1,
					r = l + 1,
					b = i;

		if (l < *n && (minheap ? h[l].d < h[b].d : h[l].d > h[b].d))
			b = l;
		if (r < *n && (minheap ? h[r].d < h[b].d : h[r].d > h[b].d))
			b = r;
		if (b == i)
			break;
		{
			TpCand		t = h[i];

			h[i] = h[b];
			h[b] = t;
		}
		i = b;
	}
	return top;
}

/* ------------------------------------------------------------------------------------------------ */
/* locked reads / writes of a node's list */

static inline float
dist(const FlatGraph *g, const int8 *q, int32 id)
{
	return tp_dist_i8(q, fg_code(g, id), fg_w2_c(g), g->d3);
}

/* copy node i's list at level l into sc->list (count first); returns the count */
static int
read_list(TpGraphCtx *ctx, int32 i, int l, int32 *out)
{
	const int32 *src = l == 0 ? fg_links0_c(ctx->g, i) : fg_links_c(ctx->g, i, l);
	int			cnt;

	if (src == NULL)
	{
		out[0] = 0;
		return 0;
	}
	if (ctx->sh)
		LWLockAcquire(&ctx->sh->locks[i], LW_SHARED);
	cnt = src[0];
	if (cnt > (l == 0 ? ctx->g->M0 : ctx->g->M))
		cnt = l == 0 ? ctx->g->M0 : ctx->g->M;
	memcpy(out + 1, src + 1, sizeof(int32) * (Size) cnt);
	out[0] = cnt;
	if (ctx->sh)
		LWLockRelease(&ctx->sh->locks[i]);
	return cnt;
}

/* hnswlib's getNeighborsByHeuristic2: in = candidates sorted by distance ascending (n_in of them),
 * keeps a candidate when it is closer to the query than to every candidate kept so far; at most M.
 * Writes the kept ids (closest first) into out, returns their number. */
static int
heuristic(const FlatGraph *g, const TpCand *in, int n_in, int M, int32 *out)
{
	int			kept = 0,
				i,
				j;

	if (n_in <= M)
	{
		for (i = 0; i < n_in; i++)
			out[i] = in[i].id;
		return n_in;
	}
	for (i = 0; i < n_in && kept < M; i++)
	{
		const int8 *c = fg_code(g, in[i].id);
		bool		good = true;

		for (j = 0; j < kept; j++)
			if (tp_dist_i8(fg_code(g, out[j]), c, fg_w2_c(g), g->d3) < in[i].d)
			{
				good = false;
				break;
			}
		if (good)
			out[kept++] = in[i].id;
	}
	return kept;
}

static int
cmp_cand(const void *a, const void *b)
{
	float		da = ((const TpCand *) a)->d,
				db = ((const TpCand *) b)->d;

	return da < db ? -1 : (da > db ? 1 : 0);
}

/*
 * Beam search at one level from entry ep for the query code q: the ef best nodes land in sc->res
 * sorted by distance ascending; returns their number.
 */
static int
search_layer(TpGraphCtx *ctx, TpGraphScratch *sc, const int8 *q, int32 ep, int l, int ef)
{
	const FlatGraph *g = ctx->g;
	int			ncand = 0,
				nres = 0;
	float		bound;
	TpCand		c0;

	if (++sc->epoch == 0)
	{
		memset(sc->visited, 0, sizeof(uint32) * (Size) sc->n_cap);
		sc->epoch = 1;
	}
	c0.d = dist(g, q, ep);
	c0.id = ep;
	sc->visited[ep] = sc->epoch;
	heap_push(sc->cand, &ncand, c0, true);
	heap_push(sc->res, &nres, c0, false);
	bound = c0.d;
	while (ncand > 0)
	{
		TpCand		cur;
		int			cnt,
					j;

		if (sc->cand[0].d > bound && nres >= ef)
			break;
		cur = heap_pop(sc->cand, &ncand, true);
		cnt = read_list(ctx, cur.id, l, sc->list);
		for (j = 1; j <= cnt; j++)
		{
			int32		v = sc->list[j];
			TpCand		cv;

			if (v < 0 || v >= g->n_cap || sc->visited[v] == sc->epoch)
				continue;
			sc->visited[v] = sc->epoch;
			cv.d = dist(g, q, v);
			cv.id = v;
			if (nres < ef || cv.d < bound)
			{
				if (ncand >= sc->cap * 4 + 15)
					continue;	/* beam overflow: skip (ef_construction bounded, never reached in practice) */
				heap_push(sc->cand, &ncand, cv, true);
				heap_push(sc->res, &nres, cv, false);
				if (nres > ef)
					heap_pop(sc->res, &nres, false);
				if (nres >= ef)
					bound = sc->res[0].d;
			}
		}
	}
	/* sort the results ascending in place (the heap array order is arbitrary) */
	qsort(sc->res, (size_t) nres, sizeof(TpCand), cmp_cand);
	return nres;
}

/* set the list of node i at level l under its exclusive lock */
static void
write_list(TpGraphCtx *ctx, int32 i, int l, const int32 *ids, int cnt)
{
	int32	   *dst = l == 0 ? fg_links0(ctx->g, i) : fg_links(ctx->g, i, l);

	if (dst == NULL)
		ereport(ERROR, (errcode(ERRCODE_INDEX_CORRUPTED), errmsg("trigpoint: node %d has no level %d list", i, l), errhint("Run trigpoint_check(<index>, true); REINDEX rebuilds the index.")));
	if (ctx->sh)
		LWLockAcquire(&ctx->sh->locks[i], LW_EXCLUSIVE);
	memcpy(dst + 1, ids, sizeof(int32) * (Size) cnt);
	dst[0] = cnt;
	if (ctx->sh)
		LWLockRelease(&ctx->sh->locks[i]);
}

/* add `pos` to neighbour s's list at level l: append when there is room, else re-select with the heuristic */
static void
connect_back(TpGraphCtx *ctx, TpGraphScratch *sc, int32 s, int32 pos, int l)
{
	FlatGraph  *g = ctx->g;
	int			Mcur = l == 0 ? g->M0 : g->M;
	int32	   *dst = l == 0 ? fg_links0(g, s) : fg_links(g, s, l);
	int			cnt,
				j;

	if (dst == NULL)
		return;					/* s has no such level (cannot happen for a beam result at level l) */
	if (ctx->sh)
		LWLockAcquire(&ctx->sh->locks[s], LW_EXCLUSIVE);
	cnt = dst[0];
	for (j = 1; j <= cnt; j++)
		if (dst[j] == pos)
		{
			if (ctx->sh)
				LWLockRelease(&ctx->sh->locks[s]);
			return;
		}
	if (cnt < Mcur)
	{
		dst[cnt + 1] = pos;
		dst[0] = cnt + 1;
	}
	else
	{
		const int8 *cs = fg_code(g, s);
		int			n_in = 0,
					kept;

		sc->tmp[n_in].d = tp_dist_i8(cs, fg_code(g, pos), fg_w2_c(g), g->d3);
		sc->tmp[n_in++].id = pos;
		for (j = 1; j <= cnt; j++)
		{
			sc->tmp[n_in].d = tp_dist_i8(cs, fg_code(g, dst[j]), fg_w2_c(g), g->d3);
			sc->tmp[n_in++].id = dst[j];
		}
		qsort(sc->tmp, (size_t) n_in, sizeof(TpCand), cmp_cand);
		kept = heuristic(g, sc->tmp, n_in, Mcur, sc->list);
		memcpy(dst + 1, sc->list, sizeof(int32) * (Size) kept);
		dst[0] = kept;
	}
	if (ctx->sh)
		LWLockRelease(&ctx->sh->locks[s]);
}

/*
 * Insert position `pos` (its code must be set). Deterministic level from (seed, pos), clamped to the
 * upper blocks still available; descent from the entry point through the levels above the node's,
 * then per level an ef_construction beam, hnswlib's heuristic for the node's own list (M), and the
 * back-links with pruning (M0 at level 0, M above).
 */
void
tp_graph_insert(TpGraphCtx *ctx, int pos, TpGraphScratch *sc)
{
	FlatGraph  *g = ctx->g;
	const int8 *q = fg_code(g, pos);
	int			level = fg_draw_level(g->seed, pos, g->mult);
	int32		ep,
				maxlvl;
	int			l;

	scratch_fit(sc, g);
	if (pos < 0 || pos >= g->n_cap)
		ereport(ERROR, (errcode(ERRCODE_INDEX_CORRUPTED), errmsg("trigpoint: graph position %d out of range", pos), errhint("Run trigpoint_check(<index>, true); REINDEX rebuilds the index.")));
	if (level > 0)
	{
		/* upper blocks: one per level above 0, from the bump allocator; degrade the level when exhausted */
		uint32		first;

		if (ctx->sh)
			first = pg_atomic_fetch_add_u32(&ctx->sh->up_alloc, (uint32) level);
		else
		{
			first = (uint32) g->up_used;
			g->up_used += level;
		}
		if ((int64) first + level > g->up_cap)
		{
			level = (int) Max(0, (int64) g->up_cap - (int64) first);
			if (level > 0)
				fg_upidx(g)[pos] = (int32) first;
		}
		else
			fg_upidx(g)[pos] = (int32) first;
		if (level > 0)
		{
			int32	   *blk = (int32 *) ((char *) g + g->uplinks_off) + (size_t) first * (size_t) (g->M + 1);

			memset(blk, 0, sizeof(int32) * (size_t) level * (size_t) (g->M + 1));
		}
	}
	fg_level(g)[pos] = level;
	fg_links0(g, pos)[0] = 0;

	/* entry point */
	if (ctx->sh)
		LWLockAcquire(&ctx->sh->entry_lock, LW_SHARED);
	ep = g->enterpoint;
	maxlvl = g->maxlevel;
	if (ctx->sh)
		LWLockRelease(&ctx->sh->entry_lock);
	if (ep < 0)
	{
		/* first node: becomes the entry point (racing first inserters: the lock decides) */
		if (ctx->sh)
			LWLockAcquire(&ctx->sh->entry_lock, LW_EXCLUSIVE);
		if (g->enterpoint < 0)
		{
			g->enterpoint = pos;
			g->maxlevel = level;
			if (ctx->sh)
				LWLockRelease(&ctx->sh->entry_lock);
			if (pos + 1 > g->n)
				g->n = pos + 1;
			return;
		}
		ep = g->enterpoint;
		maxlvl = g->maxlevel;
		if (ctx->sh)
			LWLockRelease(&ctx->sh->entry_lock);
	}

	/* greedy descent above the node's level */
	{
		float		curd = dist(g, q, ep);

		for (l = maxlvl; l > level; l--)
		{
			bool		changed = true;

			while (changed)
			{
				int			cnt = read_list(ctx, ep, l, sc->list),
							j;

				changed = false;
				for (j = 1; j <= cnt; j++)
				{
					int32		v = sc->list[j];
					float		d;

					if (v < 0 || v >= g->n_cap)
						continue;
					d = dist(g, q, v);
					if (d < curd)
					{
						curd = d;
						ep = v;
						changed = true;
					}
				}
			}
		}
	}
	/* per level: beam, select, connect */
	for (l = Min(level, maxlvl); l >= 0; l--)
	{
		int			nres = search_layer(ctx, sc, q, ep, l, g->efC),
					nsel,
					j,
					k = 0;

		/* the node itself may be in the beam (a concurrent inserter already linked it): drop it */
		for (j = 0; j < nres; j++)
			if (sc->res[j].id != pos)
				sc->res[k++] = sc->res[j];
		nres = k;
		if (nres == 0)
			continue;
		nsel = heuristic(g, sc->res, nres, g->M, sc->list);
		write_list(ctx, pos, l, sc->list, nsel);
		{
			int32		sel[2 * 128 + 2];	/* the selection, copied out before connect_back reuses sc->list (M <= 128) */

			memcpy(sel, sc->list, sizeof(int32) * (Size) nsel);
			for (j = 0; j < nsel; j++)
				connect_back(ctx, sc, sel[j], pos, l);
			ep = sel[0];
		}
		CHECK_FOR_INTERRUPTS();
	}
	if (level > maxlvl)
	{
		if (ctx->sh)
			LWLockAcquire(&ctx->sh->entry_lock, LW_EXCLUSIVE);
		if (level > g->maxlevel)
		{
			g->maxlevel = level;
			g->enterpoint = pos;
		}
		if (ctx->sh)
			LWLockRelease(&ctx->sh->entry_lock);
	}
	if (ctx->sh == NULL && pos + 1 > g->n)
		g->n = pos + 1;		/* shared graphs: the leader sets n after the last barrier */
}

/* ------------------------------------------------------------------------------------------------ */
/* H13: compaction of a base graph over its live nodes */

/*
 * A new private graph holding only the nodes of `src` with newpos[i] >= 0 (their new positions; the
 * ranks are increasing in i, as cascade_live_map assigns them), room for new_cap nodes. Levels and
 * lists are carried over with the ids remapped and the dead entries dropped; a node that lost entries
 * at a level is repaired with the closest live neighbours of its dead neighbours at that level (the
 * two-hop candidates, by the graph's own distance), up to what it lost. The entry point is the old one
 * when it survives, else the highest live node. The caller inserts the log's rows after this.
 */
FlatGraph *
tp_graph_compact(const FlatGraph *src, const int32 *newpos, int n_new, int new_cap)
{
	FlatGraph  *g;
	const int32 *slevel = fg_level_c(src);
	uint32	   *seen;
	uint32		epoch = 0;
	TpCand	   *cand;
	int32	   *tmp;
	int			cand_cap = src->M0 * (src->M0 + 1) + 8,
				i,
				repaired = 0;

	if (new_cap < n_new)
		new_cap = n_new;
	g = tp_graph_private_new(new_cap, src->d3, src->M, src->efC, fg_w2_c(src), src->seed);
	if (g == NULL)
		return NULL;
	memcpy(fg_w2(g), fg_w2_c(src), sizeof(float) * (size_t) src->d3);	/* verbatim: w2 is step² already */
	g->maxlevel = -1;
	g->enterpoint = -1;
	/* pass 1: nodes, levels, upper blocks, remapped lists */
	for (i = 0; i < src->n; i++)
	{
		int32		j = newpos[i];
		int			lvl,
					l;

		if (j < 0)
			continue;
		if (j >= new_cap)
			elog(ERROR, "trigpoint: compacted graph position %d beyond its capacity %d", j, new_cap);
		tp_graph_set_code(g, j, fg_code(src, i));
		lvl = slevel[i];
		if (lvl < 0)
		{
			fg_level(g)[j] = -1;	/* never graphed (a cancelled pair) */
			fg_upidx(g)[j] = -1;
			fg_links0(g, j)[0] = 0;
			continue;
		}
		if (lvl > 0)
		{
			int64		first = g->up_used;

			if (first + lvl > g->up_cap)
				lvl = (int) Max(0, (int64) g->up_cap - first);	/* out of upper blocks: degrade (as the insert does) */
			if (lvl > 0)
			{
				fg_upidx(g)[j] = (int32) first;
				g->up_used += lvl;
				memset((int32 *) ((char *) g + g->uplinks_off) + (size_t) first * (size_t) (g->M + 1), 0,
					   sizeof(int32) * (size_t) lvl * (size_t) (g->M + 1));
			}
		}
		if (lvl == 0)
			fg_upidx(g)[j] = -1;
		fg_level(g)[j] = lvl;
		for (l = 0; l <= lvl; l++)
		{
			const int32 *sl = l == 0 ? fg_links0_c(src, i) : fg_links_c(src, i, l);
			int32	   *dl = l == 0 ? fg_links0(g, j) : fg_links(g, j, l);
			int			cnt = sl ? sl[0] : 0,
						Mcur = l == 0 ? src->M0 : src->M,
						k,
						out = 0;

			if (cnt > Mcur)
				cnt = Mcur;
			for (k = 1; k <= cnt; k++)
				if (sl[k] >= 0 && sl[k] < src->n_cap && newpos[sl[k]] >= 0)
					dl[1 + out++] = newpos[sl[k]];
			dl[0] = out;
		}
		if (lvl > g->maxlevel)
		{
			g->maxlevel = lvl;
			g->enterpoint = j;
		}
		if ((i & 4095) == 0)
			CHECK_FOR_INTERRUPTS();
	}
	g->n = n_new;
	if (src->enterpoint >= 0 && src->enterpoint < src->n && newpos[src->enterpoint] >= 0 && slevel[src->enterpoint] == g->maxlevel)
		g->enterpoint = newpos[src->enterpoint];
	/* pass 2: repair the lists that lost entries with two-hop candidates through the dead neighbours */
	seen = (uint32 *) palloc0(sizeof(uint32) * (Size) src->n_cap);
	cand = (TpCand *) palloc(sizeof(TpCand) * (Size) cand_cap);
	tmp = (int32 *) palloc(sizeof(int32) * (Size) (src->M0 + 2));
	for (i = 0; i < src->n; i++)
	{
		int32		j = newpos[i];
		int			l;

		if (j < 0 || slevel[i] < 0)
			continue;
		for (l = 0; l <= slevel[i]; l++)
		{
			const int32 *sl = l == 0 ? fg_links0_c(src, i) : fg_links_c(src, i, l);
			int32	   *dl = l == 0 ? fg_links0(g, j) : fg_links(g, j, l);
			int			Mcur = l == 0 ? src->M0 : src->M,
						cnt = sl ? Min(sl[0], Mcur) : 0,
						lost = cnt - dl[0],
						ncand = 0,
						k;

			if (lost <= 0 || dl[0] >= Mcur)
				continue;
			epoch++;
			seen[i] = epoch;
			/* mark the live neighbours already in the list, by their source id */
			for (k = 1; k <= cnt; k++)
				if (sl[k] >= 0 && sl[k] < src->n_cap && newpos[sl[k]] >= 0)
					seen[sl[k]] = epoch;
			for (k = 1; k <= cnt; k++)
			{
				int32		d = sl[k];
				const int32 *dlst;
				int			dcnt,
							q;

				if (d < 0 || d >= src->n_cap || newpos[d] >= 0 || slevel[d] < l)
					continue;	/* live, or a dead node without this level */
				dlst = l == 0 ? fg_links0_c(src, d) : fg_links_c(src, d, l);
				dcnt = dlst ? Min(dlst[0], Mcur) : 0;
				for (q = 1; q <= dcnt && ncand < cand_cap; q++)
				{
					int32		x = dlst[q];

					if (x < 0 || x >= src->n_cap || newpos[x] < 0 || seen[x] == epoch)
						continue;
					seen[x] = epoch;
					cand[ncand].id = x;
					cand[ncand].d = tp_dist_i8(fg_code(src, i), fg_code(src, x), fg_w2_c(src), src->d3);
					ncand++;
				}
			}
			if (ncand == 0)
				continue;
			qsort(cand, (size_t) ncand, sizeof(TpCand), cmp_cand);
			for (k = 0; k < ncand && dl[0] < Mcur && k < lost; k++)
				dl[1 + dl[0]++] = newpos[cand[k].id];
			repaired++;
		}
		if ((i & 4095) == 0)
			CHECK_FOR_INTERRUPTS();
	}
	pfree(tmp);
	pfree(cand);
	pfree(seen);
	(void) repaired;
	return g;
}
