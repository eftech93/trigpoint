/* Validation of a flat index image (flat_image.h) and its router graph (flat_graph.h) before anything
 * reads through their offsets. Plain C, header-only, shared by the core (cascade_open_flat: the cheap
 * check on every open, so a corrupt offset is an error and not a wild read) and the extension
 * (trigpoint_check(regclass, deep): the same plus the O(n) checks of the arrays' contents).
 *
 * Returns 0 when the image passes, -1 with a message in
 * err otherwise. The deep pass needs the whole image mapped; the cheap pass reads the header, the two
 * block-offset tables and the graph header only.
 */
#ifndef TRIGPOINT_FLAT_CHECK_H
#define TRIGPOINT_FLAT_CHECK_H

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "flat_graph.h"
#include "flat_image.h"

static inline int
tpfl_fail(char *err, size_t errlen, const char *what)
{
	if (err && errlen)
	{
		strncpy(err, what, errlen - 1);
		err[errlen - 1] = '\0';
	}
	return -1;
}

/* [off, off + bytes) inside [0, total)? (overflow-safe) */
static inline int
tpfl_range_ok(uint64_t off, uint64_t bytes, uint64_t total)
{
	return off <= total && bytes <= total - off;
}

#define TPFL_SECTION(name, off, bytes) \
	do { if (!tpfl_range_ok((off), (bytes), h->total_bytes)) { \
		snprintf(msg, sizeof msg, "flat image: section %s [" "%llu" " + " "%llu" "] outside the image (" "%llu" " bytes)", \
				 (name), (unsigned long long) (off), (unsigned long long) (bytes), (unsigned long long) h->total_bytes); \
		return tpfl_fail(err, errlen, msg); } } while (0)

/* the router graph area: header, capacities and the offsets fg_layout would have produced */
static inline int
tpfl_check_graph(const FlatGraph *g, uint64_t bytes, int32_t n, int32_t d3, int deep, char *err, size_t errlen)
{
	FlatGraph	ref;
	char		msg[200];

	if (bytes < sizeof(FlatGraph))
		return tpfl_fail(err, errlen, "flat graph: area shorter than its header");
	if (g->magic != FG_MAGIC || g->version != FG_VERSION)
		return tpfl_fail(err, errlen, "flat graph: bad magic or version");
	if (g->n_cap < 1 || g->n < 0 || g->n > g->n_cap || g->n_cap < n)
		return tpfl_fail(err, errlen, "flat graph: node counts (n, n_cap) inconsistent with the image");
	if (g->d3 != d3 || g->M < 1 || g->M > 4096 || g->M0 != 2 * g->M)
		return tpfl_fail(err, errlen, "flat graph: degree or code width inconsistent with the image");
	if (g->up_cap < 0 || g->up_used < 0 || g->up_used > g->up_cap)
		return tpfl_fail(err, errlen, "flat graph: upper-block counts inconsistent");
	if (g->maxlevel < -1 || g->maxlevel > 64 || g->enterpoint < -1 || g->enterpoint >= g->n_cap)
		return tpfl_fail(err, errlen, "flat graph: maxlevel or enterpoint out of range");	/* -1 / -1: no node yet (H14) */
	memset(&ref, 0, sizeof ref);
	fg_layout(&ref, g->n_cap, g->d3, g->M, g->up_cap);
	if (ref.codes_off != g->codes_off || ref.sc_off != g->sc_off || ref.w2_off != g->w2_off || ref.level_off != g->level_off ||
		ref.links0_off != g->links0_off || ref.upidx_off != g->upidx_off || ref.uplinks_off != g->uplinks_off ||
		ref.total_bytes != g->total_bytes || g->total_bytes != bytes)
		return tpfl_fail(err, errlen, "flat graph: section offsets do not match its capacities");
	if (!deep)
		return 0;
	{
		const int32_t *level = fg_level_c(g);
		const int32_t *upidx = fg_upidx_c(g);
		const float *w2 = fg_w2_c(g);
		int32_t		i,
					l;

		for (i = 0; i < d3; i++)
			if (!(w2[i] >= 0.0f) || !isfinite(w2[i]))
				return tpfl_fail(err, errlen, "flat graph: a level weight is negative or not finite");
		for (i = 0; i < g->n; i++)
		{
			const int32_t *l0 = fg_links0_c(g, i);

			if (level[i] == -1)
			{
				/* a position without a node: tombstoned at image time, or a cancelled pair the replay never graphed */
				if (upidx[i] != -1)
				{
					snprintf(msg, sizeof msg, "flat graph: ungraphed node %d has upper block %d", i, upidx[i]);
					return tpfl_fail(err, errlen, msg);
				}
				continue;
			}
			if (level[i] < -1 || level[i] > g->maxlevel)
			{
				snprintf(msg, sizeof msg, "flat graph: node %d has level %d (maxlevel %d)", i, level[i], g->maxlevel);
				return tpfl_fail(err, errlen, msg);
			}
			if (l0[0] < 0 || l0[0] > g->M0)
			{
				snprintf(msg, sizeof msg, "flat graph: node %d has %d level-0 links (max %d)", i, l0[0], g->M0);
				return tpfl_fail(err, errlen, msg);
			}
			for (l = 1; l <= l0[0]; l++)
				if (l0[l] < 0 || l0[l] >= g->n_cap)
				{
					snprintf(msg, sizeof msg, "flat graph: node %d links to %d outside [0, %d)", i, l0[l], g->n_cap);
					return tpfl_fail(err, errlen, msg);
				}
			if (level[i] > 0)
			{
				if (upidx[i] < 0 || (int64_t) upidx[i] + level[i] - 1 >= g->up_used)
				{
					snprintf(msg, sizeof msg, "flat graph: node %d (level %d) has upper block %d of %d", i, level[i], upidx[i], g->up_used);
					return tpfl_fail(err, errlen, msg);
				}
				for (l = 1; l <= level[i]; l++)
				{
					const int32_t *ul = fg_links_c(g, i, l);
					int32_t		k;

					if (ul[0] < 0 || ul[0] > g->M)
					{
						snprintf(msg, sizeof msg, "flat graph: node %d has %d links at level %d (max %d)", i, ul[0], l, g->M);
						return tpfl_fail(err, errlen, msg);
					}
					for (k = 1; k <= ul[0]; k++)
						if (ul[k] < 0 || ul[k] >= g->n_cap)
						{
							snprintf(msg, sizeof msg, "flat graph: node %d links to %d at level %d outside [0, %d)", i, ul[k], l, g->n_cap);
							return tpfl_fail(err, errlen, msg);
						}
				}
			}
		}
	}
	return 0;
}

/*
 * The image. deep = 0: header, section ranges, block-offset tables, graph header (what every open does).
 * deep = 1: also every array's contents (perm / pos_of as inverse maps, cells in range, norms finite,
 * calibration tables sorted, every graph link in range).
 */
static inline int
tpfl_check(const void *buf, size_t len, int deep, char *err, size_t errlen)
{
	const FlatHeader *h = (const FlatHeader *) buf;
	char		msg[200];
	int32_t		d3,
				l;
	const uint64_t *bco,
			   *bpo;

	if (!buf || len < sizeof(FlatHeader))
		return tpfl_fail(err, errlen, "flat image: shorter than its header");
	if (h->magic != TPFL_MAGIC || h->version != TPFL_VERSION)
		return tpfl_fail(err, errlen, "flat image: bad magic or version");
	if (h->total_bytes < sizeof(FlatHeader) || h->total_bytes > len)
		return tpfl_fail(err, errlen, "flat image: total_bytes outside the buffer");
	if (h->n < 0 || h->D < 1 || h->D > 65536 || h->C < 1 || h->C > (h->n > 0 ? h->n : 1) || h->L < 1 || h->L > 4)
		return tpfl_fail(err, errlen, "flat image: n, D, C or L out of range");	/* n = 0: an index built over no rows (H14) */
	for (l = 0; l < h->L; l++)
		if (h->lv[l] < 1 || h->lv[l] > h->D || (l > 0 && h->lv[l] <= h->lv[l - 1]))
			return tpfl_fail(err, errlen, "flat image: level widths not increasing within [1, D]");
	d3 = h->lv[h->L - 1];
	if (h->npos_of < h->n || h->M < 1 || h->router_mode < 0 || h->router_mode > 2)
		return tpfl_fail(err, errlen, "flat image: npos_of, M or router_mode out of range");
	if (h->n_inserted < 0 || h->n_deleted < 0 || h->n_compacted < 0)
		return tpfl_fail(err, errlen, "flat image: negative counters");
	for (l = 0; l < h->L; l++)
		if (!isfinite(h->ch[l]))
			return tpfl_fail(err, errlen, "flat image: a calibrated cosine is not finite");
	TPFL_SECTION("lo", h->lo, (uint64_t) d3 * 4);
	TPFL_SECTION("step", h->step, (uint64_t) d3 * 4);
	TPFL_SECTION("proj", h->proj, (uint64_t) d3 * (uint64_t) h->D * 4);
	TPFL_SECTION("mean", h->mean, (uint64_t) h->D * 4);
	TPFL_SECTION("cent", h->cent, (uint64_t) h->C * (uint64_t) d3 * 4);
	for (l = 0; l < h->L; l++)
	{
		if (h->cal_n[l] > h->cal_cap[l])
			return tpfl_fail(err, errlen, "flat image: a calibration table holds more entries than its capacity");
		TPFL_SECTION("cal", h->cal[l], h->cal_cap[l] * 4);
		TPFL_SECTION("err", h->err[l], (uint64_t) h->n * 4);
		TPFL_SECTION("tail", h->tail[l], (uint64_t) h->n * 4);
	}
	TPFL_SECTION("perm", h->perm, (uint64_t) h->n * 4);
	TPFL_SECTION("pos_of", h->pos_of, (uint64_t) h->npos_of * 4);
	TPFL_SECTION("cell_of", h->cell_of, (uint64_t) h->n * 4);
	TPFL_SECTION("deleted", h->deleted, (uint64_t) h->n);
	TPFL_SECTION("blk_code_off", h->blk_code_off, (uint64_t) (h->C + 1) * 8);
	TPFL_SECTION("blk_pos_off", h->blk_pos_off, (uint64_t) (h->C + 1) * 8);
	bco = FLAT_CPTR(h, h->blk_code_off, uint64_t);
	bpo = FLAT_CPTR(h, h->blk_pos_off, uint64_t);
	if (bco[0] != 0 || bpo[0] != 0)
		return tpfl_fail(err, errlen, "flat image: block offset tables do not start at 0");
	{
		int32_t		c;

		for (c = 0; c < h->C; c++)
			if (bco[c + 1] < bco[c] || bpo[c + 1] < bpo[c])
				return tpfl_fail(err, errlen, "flat image: block offset tables not monotone");
	}
	if (bpo[h->C] > (uint64_t) h->n)
		return tpfl_fail(err, errlen, "flat image: blocked positions exceed n");
	TPFL_SECTION("blk_codes", h->blk_codes, bco[h->C]);
	TPFL_SECTION("blk_pos", h->blk_pos, bpo[h->C] * 4);
	TPFL_SECTION("graph", h->graph, h->graph_bytes);
	if (tpfl_check_graph(FLAT_CPTR(h, h->graph, FlatGraph), h->graph_bytes, h->n, d3, deep, err, errlen) < 0)
		return -1;
	if (!deep)
		return 0;
	{
		const int32_t *perm = FLAT_CPTR(h, h->perm, int32_t);
		const int32_t *pos_of = FLAT_CPTR(h, h->pos_of, int32_t);
		const int32_t *cell_of = FLAT_CPTR(h, h->cell_of, int32_t);
		const int32_t *bpos = FLAT_CPTR(h, h->blk_pos, int32_t);
		const float *lo = FLAT_CPTR(h, h->lo, float);
		const float *step = FLAT_CPTR(h, h->step, float);
		int32_t		p,
					i;

		for (i = 0; i < d3; i++)
			if (!isfinite(lo[i]) || !isfinite(step[i]) || step[i] <= 0.0f)
				return tpfl_fail(err, errlen, "flat image: quantisation lo/step not finite or step <= 0");
		{
			/* P5a: the model's floats. A corrupted PCA entry, mean or centroid gives wrong answers without any
			 * structural sign; finiteness is all that can be asserted (the rows of proj are orthonormal only when
			 * the sample's covariance has full rank: a degenerate build, e.g. identical rows, keeps other rows) */
			const float *proj = FLAT_CPTR(h, h->proj, float);
			const float *mean = FLAT_CPTR(h, h->mean, float);
			const float *cent = FLAT_CPTR(h, h->cent, float);

			for (i = 0; i < d3 * h->D; i++)
				if (!isfinite(proj[i]))
					return tpfl_fail(err, errlen, "flat image: PCA projection not finite");
			for (i = 0; i < h->D; i++)
				if (!isfinite(mean[i]))
					return tpfl_fail(err, errlen, "flat image: mean not finite");
			for (i = 0; i < h->C * d3; i++)
				if (!isfinite(cent[i]))
					return tpfl_fail(err, errlen, "flat image: a centroid is not finite");
		}
		for (p = 0; p < h->n; p++)
		{
			int32_t		id = perm[p];

			if (id < 0 || id >= h->npos_of)
			{
				snprintf(msg, sizeof msg, "flat image: perm[%d] = %d outside [0, %d)", p, id, h->npos_of);
				return tpfl_fail(err, errlen, msg);
			}
			if (pos_of[id] != p)
			{
				snprintf(msg, sizeof msg, "flat image: pos_of[perm[%d] = %d] = %d, not %d", p, id, pos_of[id], p);
				return tpfl_fail(err, errlen, msg);
			}
			if (cell_of[p] < 0 || cell_of[p] >= h->C)
			{
				snprintf(msg, sizeof msg, "flat image: cell_of[%d] = %d outside [0, %d)", p, cell_of[p], h->C);
				return tpfl_fail(err, errlen, msg);
			}
		}
		for (i = 0; i < h->npos_of; i++)
			if (pos_of[i] < -1 || pos_of[i] >= h->n || (pos_of[i] >= 0 && perm[pos_of[i]] != i))
			{
				snprintf(msg, sizeof msg, "flat image: pos_of[%d] = %d is not the inverse of perm", i, pos_of[i]);
				return tpfl_fail(err, errlen, msg);
			}
		for (i = 0; (uint64_t) i < bpo[h->C]; i++)
			if (bpos[i] < 0 || bpos[i] >= h->n)
			{
				snprintf(msg, sizeof msg, "flat image: blocked position %d = %d outside [0, %d)", i, bpos[i], h->n);
				return tpfl_fail(err, errlen, msg);
			}
		for (l = 0; l < h->L; l++)
		{
			const float *e = FLAT_CPTR(h, h->err[l], float);
			const float *t = FLAT_CPTR(h, h->tail[l], float);
			const float *cal = FLAT_CPTR(h, h->cal[l], float);
			uint64_t	k;

			for (p = 0; p < h->n; p++)
				if (!isfinite(e[p]) || e[p] < 0.0f || !isfinite(t[p]) || t[p] < 0.0f)
				{
					snprintf(msg, sizeof msg, "flat image: norm of position %d at level %d negative or not finite", p, l);
					return tpfl_fail(err, errlen, msg);
				}
			for (k = 0; k < h->cal_n[l]; k++)
				if (!isfinite(cal[k]) || (k > 0 && cal[k] < cal[k - 1]))
				{
					snprintf(msg, sizeof msg, "flat image: calibration table of level %d not finite or not sorted at %llu", l, (unsigned long long) k);
					return tpfl_fail(err, errlen, msg);
				}
		}
	}
	return 0;
}

#undef TPFL_SECTION
#endif
