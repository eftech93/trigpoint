/* Trigpoint access method: the parallel CREATE INDEX (E2).
 *
 * pgvector's hnswbuild pattern: the leader creates a ParallelContext, one DSM area holds everything the
 * build touches — the vectors (filled by a parallel heap scan, each participant claiming row slots), the
 * TIDs, the heads, the labels, per-participant reduction buffers and the index image being assembled in
 * place (flat_image.h; the router graph built in it by tp_graph.c with per-node LWLocks in a separate
 * shared array) — and the participants (workers + the leader) move through the phases separated by a
 * Barrier; the participant the barrier elects does the serial steps between them (reductions, the PCA
 * solve, centroid updates, the layout, the calibration sort). Every numeric step is a range-based core
 * entry point over the DSM arrays (trigpoint_core.h "range-based build phases"); the C++ boundary rules
 * hold: no PostgreSQL call inside a core frame, interrupts polled between core calls in C.
 *
 * Determinism: the reloption `seed` fixes the PCA sample, the k-means init and every node's level, so two
 * builds give identical cells, codes, norms and calibration; the graph's links are identical for a
 * leader-only build and timing-dependent (quality-equivalent) with workers, as pgvector's.
 *
 * Memory: the area is sized to need and refused when it exceeds maintenance_work_mem (hint: that setting
 * and the container's shm_size — the area is a POSIX shm file under dynamic_shared_memory_type = posix).
 *
 * Inserts committed during a build: a plain CREATE INDEX holds ShareLock on the heap (none can); CREATE
 * INDEX CONCURRENTLY collects them through validate_index -> aminsert -> the log; a compaction cannot
 * overlap (its ShareUpdateExclusiveLock conflicts). Participants never touch the meta page or the log.
 */
#include "trigpoint.h"
#include "utils/lsyscache.h"

#include <math.h>

#include "access/parallel.h"
#include "access/table.h"
#include "access/tableam.h"
#include "access/xact.h"
#include "catalog/index.h"
#include "miscadmin.h"
#include "optimizer/optimizer.h"
#include "pgstat.h"
#include "storage/barrier.h"
#include "storage/bufmgr.h"
#include "storage/lwlock.h"
#include "tcop/tcopprot.h"
#include "utils/guc.h"
#include "utils/memutils.h"
#include "utils/snapmgr.h"
#include "utils/wait_event.h"
#include "portability/instr_time.h"

#include "extension/vector/vector.h"

#include "flat_image.h"
#include "tp_graph.h"

#define PARALLEL_KEY_TP_SHARED		UINT64CONST(0xB000000000000001)
#define PARALLEL_KEY_TP_AREA		UINT64CONST(0xB000000000000002)
#define PARALLEL_KEY_QUERY_TEXT		UINT64CONST(0xB000000000000003)

#define TP_CAL_K 10						/* the core's CAL_K */
#define TP_GRAPH_CHUNK 256				/* positions claimed per graph-insert task */

typedef struct TpBuildShared
{
	/* immutable */
	Oid			heaprelid;
	Oid			indexrelid;
	bool		isconcurrent;
	int			n_cap;			/* rows counted before the area was sized */
	int			D;
	int			metric;			/* H11 */
	int			C;
	int			L;
	int			lv[4];
	int			d3;
	int			M;
	int			efC;
	int			sample_n;
	int			train_n;
	int			n_cal;
	int			km_iters;
	int			power_iters;
	uint64		seed;
	int			nparticipants;	/* workers launched + the leader */
	int			tranche;		/* LWLock tranche of the graph locks */
	/* synchronisation */
	Barrier		barrier;
	pg_atomic_uint32 attached;	/* participants that have attached to the barrier */
	pg_atomic_uint32 next_row;	/* row slots handed out by the scan */
	pg_atomic_uint32 overflow;	/* P5e: rows the build scan saw beyond n_cap (CONCURRENTLY: the validate phase inserts them) */
	pg_atomic_uint32 next_task;	/* work units of the current phase */
	pg_atomic_uint32 failed;
	/* results of the serial steps the participants read after the barrier */
	int			n;				/* rows scanned */
	int			sample_m;
	int			train_T;
	int			cal_m;
	float		var;
	double		trace;
	int			pca_r;
	double		reltuples;
	/* area layout (byte offsets from the area start) */
	uint64		x_off,
				tids_off,
				H_off,
				labels_off,
				cov_off,
				cov_stride,
				mean_off,
				proj_off,
				lo_off,
				step_off,
				minmax_off,
				cent_off,
				ksum_off,
				kcnt_off,
				klab_off,
				sample_off,
				train_off,
				cal_idx_off,
				cal_out_off,
				cal_cnt_off,
				cell_start_off,
				order_off,
				Q_off,
				Y_off,
				img_off,
				img_bytes,
				locks_off,
				total;
	/* the parallel heap scan descriptor follows (BUFFERALIGNed) */
} TpBuildShared;

#define ParallelTableScanFromTpShared(s) ((ParallelTableScanDesc) ((char *) (s) + BUFFERALIGN(sizeof(TpBuildShared))))

typedef struct TpBuildLeader
{
	ParallelContext *pcxt;
	Snapshot	snapshot;
	TpBuildShared *shared;
	char	   *area;
} TpBuildLeader;

/* per-participant state during the scan */
typedef struct TpScanState
{
	TpBuildShared *sh;
	char	   *area;
	MemoryContext tmpctx;
} TpScanState;

/* ------------------------------------------------------------------------------------------------ */
/* helpers */

static const uint64 *sort_tids;
static int
cmp_by_tid(const void *a, const void *b)
{
	uint64		x = sort_tids[*(const int *) a],
				y = sort_tids[*(const int *) b];

	return x < y ? -1 : (x > y ? 1 : 0);
}

/* order[0..n) = slots sorted by their TID (ids are then ranks in TID order) */
static void
tp_sort_by_tid(int *order, const uint64 *tids, int n)
{
	sort_tids = tids;
	qsort(order, (size_t) n, sizeof(int), cmp_by_tid);
	sort_tids = NULL;
}

#define A(area, off, T) ((T *) ((char *) (area) + (off)))

static inline void
range_of(int total, int nparts, int p, int *lo, int *hi)
{
	*lo = (int) (((int64) total * p) / nparts);
	*hi = (int) (((int64) total * (p + 1)) / nparts);
}

static int
participant_slot(TpBuildShared *sh)
{
	return IsParallelWorker() ? ParallelWorkerNumber : sh->nparticipants - 1;
}

/* arrive at the barrier; returns true for the participant elected to do the serial step that follows */
static bool
phase_end(TpBuildShared *sh)
{
	bool		elected;

	CHECK_FOR_INTERRUPTS();
	elected = BarrierArriveAndWait(&sh->barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);
	if (pg_atomic_read_u32(&sh->failed))
		ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("trigpoint: parallel build failed in another participant"),
						errhint("The participant's own error is in the server log.")));
	return elected;
}

static void
count_callback(Relation index, ItemPointer tid, Datum *values, bool *isnull, bool tupleIsAlive, void *state)
{
	if (!isnull[0])
		(*(int64 *) state)++;
}

static void
scan_callback(Relation index, ItemPointer tid, Datum *values, bool *isnull, bool tupleIsAlive, void *state)
{
	TpScanState *st = (TpScanState *) state;
	TpBuildShared *sh = st->sh;
	MemoryContext old;
	Vector	   *v;
	uint32		slot;
	uint64		t;

	if (isnull[0])
		return;
	slot = pg_atomic_fetch_add_u32(&sh->next_row, 1);
	if (slot >= (uint32) sh->n_cap)
	{
		/* Rows committed between the counting scan's snapshot and the build scan's. Under CONCURRENTLY the table is not
		 * locked, so a writer can add rows meanwhile; the area has no slot for them, and the validate phase of the
		 * concurrent build inserts every heap row the index lacks (through aminsert, into the log), so they are left to
		 * it. A plain build holds ShareLock on the table and cannot see this. */
		if (sh->isconcurrent)
		{
			pg_atomic_fetch_add_u32(&sh->overflow, 1);
			return;
		}
		ereport(ERROR, (errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
						errmsg("trigpoint: the heap grew between the counting scan and the build scan (%u rows, %d counted)", slot + 1, sh->n_cap),
						errhint("Rows were inserted while the index was being built; run the CREATE INDEX again.")));
	}
	old = MemoryContextSwitchTo(st->tmpctx);
	v = DatumGetVector(values[0]);
	if (v->dim != sh->D)
		ereport(ERROR, (errcode(ERRCODE_DATA_EXCEPTION),
						errmsg("trigpoint: vector has %d dimensions, index column expects %d", v->dim, sh->D)));
	memcpy(A(st->area, sh->x_off, float) + (size_t) slot * sh->D, v->x, sizeof(float) * (size_t) sh->D);
	if (sh->metric == TP_METRIC_COSINE)
		tp_normalize(A(st->area, sh->x_off, float) + (size_t) slot * sh->D, sh->D);	/* the area's copy: an inline datum points into the heap page */
	t = ((uint64) ItemPointerGetBlockNumber(tid) << 16) | (uint64) ItemPointerGetOffsetNumber(tid);
	A(st->area, sh->tids_off, uint64)[slot] = t;
	MemoryContextSwitchTo(old);
	MemoryContextReset(st->tmpctx);
}

/* ------------------------------------------------------------------------------------------------ */
/* the phases (every participant runs this; the elected one does the serial steps) */

static void
tp_participate(TpBuildShared *sh, char *area, Relation heap, Relation index, IndexInfo *indexInfo, double *phase_s)
{
	const int	P = sh->nparticipants;
	const int	p = participant_slot(sh);
	const int	D = sh->D,
				d3 = sh->d3,
				L = sh->L;
	float	   *x = A(area, sh->x_off, float);
	float	   *H = A(area, sh->H_off, float);
	int		   *labels = A(area, sh->labels_off, int);
	float	   *mean = A(area, sh->mean_off, float);
	float	   *proj = A(area, sh->proj_off, float);
	float	   *lo = A(area, sh->lo_off, float);
	float	   *step = A(area, sh->step_off, float);
	float	   *cent = A(area, sh->cent_off, float);
	int		   *sample_idx = A(area, sh->sample_off, int);
	int		   *train_idx = A(area, sh->train_off, int);
	int		   *klab = A(area, sh->klab_off, int);
	int		   *cal_idx = A(area, sh->cal_idx_off, int);
	int		   *cell_start = A(area, sh->cell_start_off, int);
	int		   *order = A(area, sh->order_off, int);	/* id (TID rank) -> slot (scan arrival) */
	FlatHeader *img = (FlatHeader *) (area + sh->img_off);
	TpGraphShared *gsh = (TpGraphShared *) (area + sh->locks_off);
	FlatGraph  *graph = NULL;
	instr_time	t_prev,
				t_now;
	int			ph = 0,
				lo_i,
				hi_i,
				i,
				l,
				it;

#define PHASE_DONE() do { if (phase_s) { INSTR_TIME_SET_CURRENT(t_now); phase_s[ph] += INSTR_TIME_GET_DOUBLE(t_now) - INSTR_TIME_GET_DOUBLE(t_prev); t_prev = t_now; } ph++; } while (0)

	INSTR_TIME_SET_CURRENT(t_prev);

	/* ---- phase 0: parallel heap scan into the area ---- */
	{
		TpScanState st;
		TableScanDesc scan;

		st.sh = sh;
		st.area = area;
		st.tmpctx = AllocSetContextCreate(CurrentMemoryContext, "trigpoint parallel build row", ALLOCSET_DEFAULT_SIZES);
		scan = table_beginscan_parallel(heap, ParallelTableScanFromTpShared(sh));
		table_index_build_scan(heap, index, indexInfo, true, true, scan_callback, &st, scan);
		MemoryContextDelete(st.tmpctx);
	}
	if (phase_end(sh))
	{
		/* serial 1: row count; ids numbered in TID order (the parallel scan fills slots in arrival order, which
		 * differs from run to run — ordering by TID makes the sample, the k-means and every id deterministic);
		 * the PCA sample (ids -> slots) and its mean */
		const uint64 *tids = A(area, sh->tids_off, uint64);

		sh->n = (int) Min(pg_atomic_read_u32(&sh->next_row), (uint32) sh->n_cap);
		if (pg_atomic_read_u32(&sh->overflow) > 0)
			ereport(LOG, (errmsg("trigpoint: %u rows added during the concurrent build of \"%s\" are left to its validation phase",
								 pg_atomic_read_u32(&sh->overflow), get_rel_name(sh->indexrelid))));
		if (sh->n < 1)
		{
			/* H14: no rows -- every participant leaves after this barrier and the leader writes the trivial model */
			phase_end(sh);
			return;
		}
		if (sh->C > sh->n)
			sh->C = sh->n;
		for (i = 0; i < sh->n; i++)
			order[i] = i;
		tp_sort_by_tid(order, tids, sh->n);
		sh->sample_m = tp_sample(sh->seed, sh->n, Min(sh->sample_n, sh->n), sample_idx);
		for (i = 0; i < sh->sample_m; i++)
			sample_idx[i] = order[sample_idx[i]];	/* the sampled ids as slots */
		tp_mean(x, sample_idx, sh->sample_m, D, mean);
	}
	phase_end(sh);
	PHASE_DONE();				/* 0: scan */
	if (sh->n < 1)
		return;					/* H14: an empty table (the leader saw it in serial 1) */

	/* ---- phase 1: covariance partials over the sample ---- */
	{
		double	   *cov = (double *) (area + sh->cov_off + (uint64) p * sh->cov_stride);

		memset(cov, 0, sizeof(double) * (size_t) D * D);
		range_of(sh->sample_m, P, p, &lo_i, &hi_i);
		if (tp_cov_partial(x, sample_idx, lo_i, hi_i, D, mean, cov) != 0)
			ereport(ERROR, (errcode(ERRCODE_OUT_OF_MEMORY), errmsg("trigpoint: %s", tp_last_error())));
	}
	if (phase_end(sh))
	{
		/* serial 2a: reduce the partials, normalise, draw Q */
		double	   *cov0 = (double *) (area + sh->cov_off);
		size_t		k;

		for (i = 1; i < P; i++)
		{
			const double *ci = (const double *) (area + sh->cov_off + (uint64) i * sh->cov_stride);

			for (k = 0; k < (size_t) D * D; k++)
				cov0[k] += ci[k];
		}
		sh->trace = tp_pca_prepare(cov0, sh->sample_m, D);
		sh->pca_r = tp_pca_rank(D, d3);
		if (tp_pca_init((double *) (area + sh->Q_off), D, sh->pca_r, sh->seed) != 0)
			ereport(ERROR, (errcode(ERRCODE_OUT_OF_MEMORY), errmsg("trigpoint: %s", tp_last_error())));
	}
	phase_end(sh);
	/* block subspace iteration: the participants multiply column ranges, the elected one re-orthonormalises */
	for (it = 0; it <= sh->power_iters; it++)
	{
		const double *cov0 = (const double *) (area + sh->cov_off);
		double	   *Q = (double *) (area + sh->Q_off),
				   *Y = (double *) (area + sh->Y_off);

		range_of(sh->pca_r, P, p, &lo_i, &hi_i);
		tp_pca_mul_partial(cov0, D, Q, lo_i, hi_i, Y);
		if (phase_end(sh))
		{
			if (it < sh->power_iters)
			{
				memcpy(Q, Y, sizeof(double) * (size_t) sh->pca_r * D);
				if (tp_orthonormalise(Q, D, sh->pca_r) != 0)
					ereport(ERROR, (errcode(ERRCODE_OUT_OF_MEMORY), errmsg("trigpoint: %s", tp_last_error())));
			}
			else
			{
				sh->var = tp_pca_finish(Q, Y, D, sh->pca_r, d3, sh->trace, proj);
				if (sh->var < 0)
					ereport(ERROR, (errcode(ERRCODE_OUT_OF_MEMORY), errmsg("trigpoint: %s", tp_last_error())));
			}
		}
		phase_end(sh);
	}
	PHASE_DONE();				/* 1: fit */

	/* ---- phase 2: project every row, per-dimension range ---- */
	{
		float	   *mn = A(area, sh->minmax_off, float) + (size_t) p * 2 * d3;
		float	   *mx = mn + d3;

		for (i = 0; i < d3; i++)
		{
			mn[i] = INFINITY;
			mx[i] = -INFINITY;
		}
		range_of(sh->n, P, p, &lo_i, &hi_i);
		if (tp_project_rows(x, lo_i, hi_i, D, d3, proj, mean, H) != 0)
			ereport(ERROR, (errcode(ERRCODE_OUT_OF_MEMORY), errmsg("trigpoint: %s", tp_last_error())));
		tp_minmax_partial(H, lo_i, hi_i, d3, mn, mx);
	}
	if (phase_end(sh))
	{
		/* serial 3: lo / step, the k-means training set and init */
		float	   *mn0 = A(area, sh->minmax_off, float);
		float	   *mx0 = mn0 + d3;

		for (i = 1; i < P; i++)
		{
			const float *mni = A(area, sh->minmax_off, float) + (size_t) i * 2 * d3;

			for (l = 0; l < d3; l++)
			{
				mn0[l] = Min(mn0[l], mni[l]);
				mx0[l] = Max(mx0[l], mni[d3 + l]);
			}
		}
		tp_lo_step(mn0, mx0, d3, lo, step);
		sh->train_T = tp_sample(sh->seed + 1, sh->n, Min(sh->train_n, sh->n), train_idx);
		for (i = 0; i < sh->train_T; i++)
			train_idx[i] = order[train_idx[i]];	/* ids -> slots */
		if (tp_kmeans_init(H, train_idx, sh->train_T, d3, sh->C, sh->seed, cent) != 0)
			ereport(ERROR, (errcode(ERRCODE_OUT_OF_MEMORY), errmsg("trigpoint: %s", tp_last_error())));
	}
	phase_end(sh);
	PHASE_DONE();				/* 2: head */

	/* ---- phase 3: Lloyd iterations on the training rows, then every row's label ---- */
	for (it = 0; it < sh->km_iters; it++)
	{
		double	   *ksum = (double *) (area + sh->ksum_off) + (size_t) p * sh->C * d3;
		int		   *kcnt = A(area, sh->kcnt_off, int) + (size_t) p * sh->C;

		memset(ksum, 0, sizeof(double) * (size_t) sh->C * d3);
		memset(kcnt, 0, sizeof(int) * (size_t) sh->C);
		range_of(sh->train_T, P, p, &lo_i, &hi_i);
		if (tp_kmeans_assign_partial(H, train_idx, lo_i, hi_i, d3, cent, sh->C, klab, ksum, kcnt) != 0)
			ereport(ERROR, (errcode(ERRCODE_OUT_OF_MEMORY), errmsg("trigpoint: %s", tp_last_error())));
		if (phase_end(sh))
		{
			double	   *s0 = (double *) (area + sh->ksum_off);
			int		   *c0 = A(area, sh->kcnt_off, int);
			size_t		k;

			for (i = 1; i < P; i++)
			{
				const double *si = (const double *) (area + sh->ksum_off) + (size_t) i * sh->C * d3;
				const int  *ci = A(area, sh->kcnt_off, int) + (size_t) i * sh->C;

				for (k = 0; k < (size_t) sh->C * d3; k++)
					s0[k] += si[k];
				for (k = 0; k < (size_t) sh->C; k++)
					c0[k] += ci[k];
			}
			if (tp_kmeans_update(H, train_idx, sh->train_T, d3, sh->C, s0, c0, sh->seed, it, cent) != 0)
				ereport(ERROR, (errcode(ERRCODE_OUT_OF_MEMORY), errmsg("trigpoint: %s", tp_last_error())));
		}
		phase_end(sh);
	}
	range_of(sh->n, P, p, &lo_i, &hi_i);
	if (tp_kmeans_assign_partial(H, NULL, lo_i, hi_i, d3, cent, sh->C, labels, NULL, NULL) != 0)
		ereport(ERROR, (errcode(ERRCODE_OUT_OF_MEMORY), errmsg("trigpoint: %s", tp_last_error())));
	if (phase_end(sh))
	{
		/* serial 5: cell layout, the image's offsets and small arrays, the graph area */
		int		   *cnt = (int *) palloc0(sizeof(int) * (Size) sh->C);
		int		   *fill = (int *) palloc(sizeof(int) * (Size) sh->C);
		uint64		cal_cap[4] = {0, 0, 0, 0};
		int32	   *perm,
				   *pos_of,
				   *cell_of;
		uint64	   *bco,
				   *bpo;
		int			c;

		for (i = 0; i < sh->n; i++)
			cnt[labels[order[i]]]++;
		cell_start[0] = 0;
		for (c = 0; c < sh->C; c++)
			cell_start[c + 1] = cell_start[c] + cnt[c];
		sh->cal_m = Min(sh->n_cal, sh->n);
		for (l = 0; l < L; l++)
			cal_cap[l] = (uint64) sh->cal_m * TP_CAL_K;
		{
			long long	need = tp_flat_layout(img, sh->n, D, sh->C, L, sh->lv, sh->M, sh->efC, sh->n, cal_cap, cnt,
											   (uint64) tp_graph_bytes(sh->n, d3, sh->M));

			if (need < 0 || (uint64) need > sh->img_bytes)
				elog(ERROR, "trigpoint: image layout (%lld bytes) exceeds the reserved region (" UINT64_FORMAT ")", need, sh->img_bytes);
		}
		img->var_captured = sh->var;
		memcpy(FLAT_PTR(img, img->lo, float), lo, sizeof(float) * (size_t) d3);
		memcpy(FLAT_PTR(img, img->step, float), step, sizeof(float) * (size_t) d3);
		memcpy(FLAT_PTR(img, img->proj, float), proj, sizeof(float) * (size_t) d3 * D);
		memcpy(FLAT_PTR(img, img->mean, float), mean, sizeof(float) * (size_t) D);
		memcpy(FLAT_PTR(img, img->cent, float), cent, sizeof(float) * (size_t) sh->C * d3);
		perm = FLAT_PTR(img, img->perm, int32);
		pos_of = FLAT_PTR(img, img->pos_of, int32);
		cell_of = FLAT_PTR(img, img->cell_of, int32);
		memcpy(fill, cell_start, sizeof(int) * (Size) sh->C);
		for (i = 0; i < sh->n; i++)
		{
			const int	lab = labels[order[i]];
			int			pos = fill[lab]++;

			perm[pos] = i;			/* ids in TID order */
			pos_of[i] = pos;
			cell_of[pos] = lab;
		}
		bco = FLAT_PTR(img, img->blk_code_off, uint64);
		bpo = FLAT_PTR(img, img->blk_pos_off, uint64);
		for (c = 0; c <= sh->C; c++)
		{
			bco[c] = (uint64) cell_start[c] * d3;
			bpo[c] = (uint64) cell_start[c];
		}
		{
			int32	   *bpos = FLAT_PTR(img, img->blk_pos, int32);

			for (i = 0; i < sh->n; i++)
				bpos[i] = i;		/* a cell's positions are contiguous: pos = cell_start + j */
		}
		memset(FLAT_PTR(img, img->deleted, char), 0, (size_t) sh->n);
		graph = (FlatGraph *) FLAT_PTR(img, img->graph, char);
		tp_graph_init(graph, (Size) img->graph_bytes, sh->n, d3, sh->M, sh->efC, step, sh->seed);
		tp_graph_shared_init(gsh, sh->n, sh->tranche);
		pg_atomic_write_u32(&sh->next_task, 0);
		pfree(cnt);
		pfree(fill);
	}
	phase_end(sh);
	PHASE_DONE();				/* 3: kmeans */

	/* ---- phase 4: encode positions [lo, hi): codes into the cell blocks, norms, router codes ---- */
	graph = (FlatGraph *) FLAT_PTR(img, img->graph, char);
	{
		int8	   *row = (int8 *) palloc((Size) d3);
		float		e[4],
					t[4];
		const int32 *perm = FLAT_CPTR(img, img->perm, int32);
		const int32 *cell_of = FLAT_CPTR(img, img->cell_of, int32);
		int8	   *bcodes = FLAT_PTR(img, img->blk_codes, int8);
		const uint64 *bco = FLAT_CPTR(img, img->blk_code_off, uint64);

		range_of(sh->n, P, p, &lo_i, &hi_i);
		for (i = lo_i; i < hi_i; i++)
		{
			const int	id = perm[i],
						slot = order[id],
						c = cell_of[i],
						j = i - cell_start[c],
						cntc = cell_start[c + 1] - cell_start[c];
			int8	   *blk = bcodes + bco[c];

			tp_encode_row(x + (size_t) slot * D, H + (size_t) slot * d3, D, L, sh->lv, lo, step, mean, row, e, t);
			for (l = 0; l < L; l++)
			{
				const int	lo_d = l ? sh->lv[l - 1] : 0,
							w = sh->lv[l] - lo_d;

				memcpy(blk + (size_t) cntc * lo_d + (size_t) j * w, row + lo_d, (size_t) w);
				FLAT_PTR(img, img->err[l], float)[i] = e[l];
				FLAT_PTR(img, img->tail[l], float)[i] = t[l];
			}
			tp_graph_set_code(graph, i, row);
			if ((i & 4095) == 0)
				CHECK_FOR_INTERRUPTS();
		}
		pfree(row);
	}
	phase_end(sh);
	PHASE_DONE();				/* 4: codes */

	/* ---- phase 5: the router graph, positions claimed in chunks ---- */
	{
		TpGraphCtx	ctx;
		TpGraphScratch *sc = tp_graph_scratch_new(graph);

		ctx.g = graph;
		ctx.sh = gsh;
		for (;;)
		{
			uint32		task = pg_atomic_fetch_add_u32(&sh->next_task, 1);
			int			a = (int) task * TP_GRAPH_CHUNK,
						b;

			if (a >= sh->n)
				break;
			b = Min(sh->n, a + TP_GRAPH_CHUNK);
			for (i = a; i < b; i++)
				tp_graph_insert(&ctx, i, sc);
			CHECK_FOR_INTERRUPTS();
		}
		tp_graph_scratch_free(sc);
	}
	if (phase_end(sh))
	{
		/* serial 6: close the graph, pick the calibration queries */
		graph->n = sh->n;
		graph->up_used = (int32) pg_atomic_read_u32(&gsh->up_alloc);
		if (graph->up_used > graph->up_cap)
			graph->up_used = graph->up_cap;
		tp_sample(sh->seed + 2, sh->n, sh->cal_m, cal_idx);
		pg_atomic_write_u32(&sh->next_task, 0);
	}
	phase_end(sh);
	PHASE_DONE();				/* 5: graph */

	/* ---- phase 6: calibration queries over the finished image ---- */
	if (sh->cal_m > 0)
	{
		void	   *fz = tp_index_open_flat(img, (size_t) img->total_bytes);
		float	   *out = A(area, sh->cal_out_off, float) + (size_t) p * L * sh->cal_m * TP_CAL_K;
		int		   *ocnt = A(area, sh->cal_cnt_off, int) + (size_t) p * L;
		float		g[4 * TP_CAL_K];
		int			gc[4];

		if (fz == NULL)
			ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("trigpoint: cannot open the image being built: %s", tp_last_error())));
		for (l = 0; l < L; l++)
			ocnt[l] = 0;
		range_of(sh->cal_m, P, p, &lo_i, &hi_i);
		for (i = lo_i; i < hi_i; i++)
		{
			int			r = tp_calibrate_query(fz, x, D, H, order, cal_idx[i], g, gc);

			if (r < 0)
				ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("trigpoint: %s", tp_last_error())));
			if (r == 0)
				continue;
			for (l = 0; l < L; l++)
			{
				memcpy(out + (size_t) l * sh->cal_m * TP_CAL_K + ocnt[l], g + l * TP_CAL_K, sizeof(float) * (size_t) gc[l]);
				ocnt[l] += gc[l];
			}
			if ((i & 63) == 0)
				CHECK_FOR_INTERRUPTS();
		}
		tp_index_free(fz);
	}
	if (phase_end(sh))
	{
		/* serial 7: gather, sort, store the cosines and the default c_h */
		for (l = 0; l < L; l++)
		{
			float	   *dst = FLAT_PTR(img, img->cal[l], float);
			uint64		n = 0;

			for (i = 0; i < P; i++)
			{
				const float *src = A(area, sh->cal_out_off, float) + ((size_t) i * L + l) * sh->cal_m * TP_CAL_K;
				int			cnt = A(area, sh->cal_cnt_off, int)[(size_t) i * L + l];

				if (n + cnt > img->cal_cap[l])
					cnt = (int) (img->cal_cap[l] - n);
				memcpy(dst + n, src, sizeof(float) * (size_t) cnt);
				n += cnt;
			}
			if (n > 1)
				qsort(dst, (size_t) n, sizeof(float), tp_cmp_float);
			img->cal_n[l] = n;
			img->ch[l] = n ? tp_ch_from_sorted(dst, (int) n, 0.01) : 1.f;
		}
	}
	phase_end(sh);
	PHASE_DONE();				/* 6: calib */
#undef PHASE_DONE
}

/* ------------------------------------------------------------------------------------------------ */
/* worker entry */

void
tp_parallel_build_main(dsm_segment *seg, shm_toc *toc)
{
	char	   *sharedquery;
	TpBuildShared *sh;
	char	   *area;
	Relation	heap,
				index;
	IndexInfo  *indexInfo;
	LOCKMODE	heapLockmode,
				indexLockmode;

	sharedquery = shm_toc_lookup(toc, PARALLEL_KEY_QUERY_TEXT, true);
	debug_query_string = sharedquery;
	pgstat_report_activity(STATE_RUNNING, debug_query_string);
	sh = (TpBuildShared *) shm_toc_lookup(toc, PARALLEL_KEY_TP_SHARED, false);
	area = (char *) shm_toc_lookup(toc, PARALLEL_KEY_TP_AREA, false);
	if (!sh->isconcurrent)
	{
		heapLockmode = ShareLock;
		indexLockmode = AccessExclusiveLock;
	}
	else
	{
		heapLockmode = ShareUpdateExclusiveLock;
		indexLockmode = RowExclusiveLock;
	}
	heap = table_open(sh->heaprelid, heapLockmode);
	index = index_open(sh->indexrelid, indexLockmode);
	indexInfo = BuildIndexInfo(index);
	indexInfo->ii_Concurrent = sh->isconcurrent;
	LWLockRegisterTranche(sh->tranche, "trigpoint_build");
	BarrierAttach(&sh->barrier);
	pg_atomic_fetch_add_u32(&sh->attached, 1);
	PG_TRY();
	{
		tp_participate(sh, area, heap, index, indexInfo, NULL);
	}
	PG_CATCH();
	{
		pg_atomic_write_u32(&sh->failed, 1);
		PG_RE_THROW();
	}
	PG_END_TRY();
	BarrierDetach(&sh->barrier);
	index_close(index, indexLockmode);
	table_close(heap, heapLockmode);
}

/* ------------------------------------------------------------------------------------------------ */
/* the leader */

static int64
count_rows(Relation heap, Relation index, IndexInfo *indexInfo)
{
	int64		n = 0;

	table_index_build_scan(heap, index, indexInfo, true, true, count_callback, &n, NULL);
	return n;
}

/*
 * Build the index with `request` workers (plus the leader); fills meta and writes the pages. Returns
 * the number of rows. nparticipants_out = workers launched + 1.
 */
int64
tp_parallel_build(Relation heap, Relation index, IndexInfo *indexInfo, TrigpointMetaPageData *meta,
				  const TrigpointOptions *opts, int request, int *nparticipants_out, double *phase_s)
{
	TpBuildLeader *leader = (TpBuildLeader *) palloc0(sizeof(TpBuildLeader));
	TpBuildShared *sh;
	char	   *area;
	ParallelContext *pcxt = NULL;
	dsm_segment *ownseg = NULL;
	Snapshot	snapshot;
	Size		estshared,
				estarea;
	int64		n_cap;
	int			D = meta->D,
				L = meta->L,
				d3 = meta->lv[L - 1],
				C,
				P,
				querylen = 0;
	uint64		off = 0;
	uint64		img_bytes,
				graph_bytes;
	uint64		cal_cap[4];
	int			l;
	instr_time	t0,
				t1;
	bool		isconcurrent = indexInfo->ii_Concurrent;

	INSTR_TIME_SET_CURRENT(t0);
	n_cap = count_rows(heap, index, indexInfo);
	if (n_cap < 1)
	{
		/* H14: no rows -- the caller writes the trivial model */
		pfree(leader);
		*nparticipants_out = 1;
		return 0;
	}
	if (n_cap > INT32_MAX)
		ereport(ERROR, (errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED), errmsg("trigpoint: too many rows")));
	C = (opts && opts->cells > 0) ? opts->cells : tp_auto_cells(n_cap);
	if (C > n_cap)
		C = (int) n_cap;
	P = Max(1, request + 1);

	/* ---- area layout (sizes from the counted rows; the image region sized for n_cap positions) ---- */
#define RESERVE(bytes) (off = TYPEALIGN(64, off), off += (bytes), off - (bytes))
	{
		int			sample_m = Min(opts ? opts->sample : 200000, (int) n_cap),
					train_T = Min(100000, (int) n_cap),
					cal_m = Min(opts ? opts->calibration_queries : 2000, (int) n_cap),
					*dummy_cnt = (int *) palloc0(sizeof(int) * (Size) C);
		FlatHeader	tmp;

		dummy_cnt[0] = (int) n_cap;
		for (l = 0; l < 4; l++)
			cal_cap[l] = l < L ? (uint64) cal_m * TP_CAL_K : 0;
		graph_bytes = (uint64) tp_graph_bytes((int) n_cap, d3, meta->M);
		img_bytes = (uint64) tp_flat_layout(&tmp, (int) n_cap, D, C, L, meta->lv, meta->M, meta->efC, (int) n_cap, cal_cap, dummy_cnt, graph_bytes);
		pfree(dummy_cnt);

		sh = (TpBuildShared *) palloc0(sizeof(TpBuildShared));	/* a template, copied into the DSM below */
		sh->x_off = RESERVE((uint64) n_cap * D * 4);
		sh->tids_off = RESERVE((uint64) n_cap * 8);
		sh->H_off = RESERVE((uint64) n_cap * d3 * 4);
		sh->labels_off = RESERVE((uint64) n_cap * 4);
		sh->cov_stride = TYPEALIGN(64, (uint64) D * D * 8);
		sh->cov_off = RESERVE(sh->cov_stride * P);
		sh->mean_off = RESERVE((uint64) D * 4);
		sh->proj_off = RESERVE((uint64) d3 * D * 4);
		sh->lo_off = RESERVE((uint64) d3 * 4);
		sh->step_off = RESERVE((uint64) d3 * 4);
		sh->minmax_off = RESERVE((uint64) P * 2 * d3 * 4);
		sh->cent_off = RESERVE((uint64) C * d3 * 4);
		sh->ksum_off = RESERVE((uint64) P * C * d3 * 8);
		sh->kcnt_off = RESERVE((uint64) P * C * 4);
		sh->klab_off = RESERVE((uint64) train_T * 4);
		sh->sample_off = RESERVE((uint64) sample_m * 4);
		sh->train_off = RESERVE((uint64) train_T * 4);
		sh->cal_idx_off = RESERVE((uint64) cal_m * 4);
		sh->cal_out_off = RESERVE((uint64) P * L * cal_m * TP_CAL_K * 4);
		sh->cal_cnt_off = RESERVE((uint64) P * L * 4);
		sh->cell_start_off = RESERVE((uint64) (C + 1) * 4);
		sh->order_off = RESERVE((uint64) n_cap * 4);
		sh->Q_off = RESERVE((uint64) tp_pca_rank(D, d3) * D * 8);
		sh->Y_off = RESERVE((uint64) tp_pca_rank(D, d3) * D * 8);
		sh->img_bytes = img_bytes;
		sh->img_off = RESERVE(img_bytes);
		sh->locks_off = RESERVE((uint64) tp_graph_shared_bytes((int) n_cap));
		sh->total = TYPEALIGN(64, off);
		sh->sample_n = sample_m;
		sh->train_n = train_T;
		sh->n_cal = cal_m;
	}
#undef RESERVE
	if (sh->total > (uint64) maintenance_work_mem * 1024)
		ereport(ERROR, (errcode(ERRCODE_INSUFFICIENT_RESOURCES),
						errmsg("trigpoint: the parallel build needs %.1f GB of shared memory for %lld rows x %d dimensions, more than maintenance_work_mem (%d MB)",
							   (double) sh->total / 1e9, (long long) n_cap, D, maintenance_work_mem / 1024),
						errhint("Raise maintenance_work_mem to at least %lld MB (and, in a container with dynamic_shared_memory_type = posix, the container's shm_size). A build without workers (max_parallel_maintenance_workers = 0) uses the same area.",
								(long long) ((sh->total + (1 << 20) - 1) >> 20))));

	sh->heaprelid = RelationGetRelid(heap);
	sh->indexrelid = RelationGetRelid(index);
	sh->isconcurrent = isconcurrent;
	sh->n_cap = (int) n_cap;
	sh->D = D;
	sh->metric = meta->metric;
	sh->C = C;
	sh->L = L;
	for (l = 0; l < 4; l++)
		sh->lv[l] = meta->lv[l];
	sh->d3 = d3;
	sh->M = meta->M;
	sh->efC = meta->efC;
	sh->km_iters = opts ? opts->kmeans_iters : 15;
	sh->power_iters = 30;
	sh->seed = (uint64) (opts ? opts->seed : 0);
	sh->tranche = LWLockNewTrancheId();
	LWLockRegisterTranche(sh->tranche, "trigpoint_build");

	/* ---- the shared area: a ParallelContext with workers, or (leader-only) a DSM segment of our own, because a
	 * ParallelContext with 0 workers keeps its memory backend-private and palloc cannot hold an 8 GB area ---- */
	snapshot = isconcurrent ? RegisterSnapshot(GetTransactionSnapshot()) : SnapshotAny;
	estshared = add_size(BUFFERALIGN(sizeof(TpBuildShared)), table_parallelscan_estimate(heap, snapshot));
	estarea = (Size) sh->total;
	if (request > 0)
	{
		EnterParallelMode();
		pcxt = CreateParallelContext("trigpoint", "tp_parallel_build_main", request);
		shm_toc_estimate_chunk(&pcxt->estimator, estshared);
		shm_toc_estimate_chunk(&pcxt->estimator, estarea);
		shm_toc_estimate_keys(&pcxt->estimator, 2);
		if (debug_query_string)
		{
			querylen = strlen(debug_query_string);
			shm_toc_estimate_chunk(&pcxt->estimator, querylen + 1);
			shm_toc_estimate_keys(&pcxt->estimator, 1);
		}
		InitializeParallelDSM(pcxt);
		{
			TpBuildShared *dsh = (TpBuildShared *) shm_toc_allocate(pcxt->toc, estshared);

			memcpy(dsh, sh, sizeof(TpBuildShared));
			pfree(sh);
			sh = dsh;
		}
		area = (char *) shm_toc_allocate(pcxt->toc, estarea);
		shm_toc_insert(pcxt->toc, PARALLEL_KEY_TP_SHARED, sh);
		shm_toc_insert(pcxt->toc, PARALLEL_KEY_TP_AREA, area);
		if (debug_query_string)
		{
			char	   *sharedquery = (char *) shm_toc_allocate(pcxt->toc, querylen + 1);

			memcpy(sharedquery, debug_query_string, querylen + 1);
			shm_toc_insert(pcxt->toc, PARALLEL_KEY_QUERY_TEXT, sharedquery);
		}
	}
	else
	{
		Size		shared_al = TYPEALIGN(64, estshared);

		ownseg = dsm_create(shared_al + estarea, 0);
		{
			TpBuildShared *dsh = (TpBuildShared *) dsm_segment_address(ownseg);

			memcpy(dsh, sh, sizeof(TpBuildShared));
			pfree(sh);
			sh = dsh;
		}
		area = (char *) dsm_segment_address(ownseg) + shared_al;
	}
	table_parallelscan_initialize(heap, ParallelTableScanFromTpShared(sh), snapshot);
	BarrierInit(&sh->barrier, 0);
	pg_atomic_init_u32(&sh->attached, 0);
	pg_atomic_init_u32(&sh->next_row, 0);
	pg_atomic_init_u32(&sh->overflow, 0);
	pg_atomic_init_u32(&sh->next_task, 0);
	pg_atomic_init_u32(&sh->failed, 0);

	if (pcxt)
		LaunchParallelWorkers(pcxt);
	sh->nparticipants = (pcxt ? pcxt->nworkers_launched : 0) + 1;
	leader->pcxt = pcxt;
	leader->snapshot = snapshot;
	leader->shared = sh;
	leader->area = area;
	*nparticipants_out = sh->nparticipants;
	BarrierAttach(&sh->barrier);
	/* every launched worker attaches before the first phase ends, else a late one would join mid-build */
	while (pcxt && (int) pg_atomic_read_u32(&sh->attached) < pcxt->nworkers_launched)
	{
		CHECK_FOR_INTERRUPTS();
		pg_usleep(1000);
	}
	INSTR_TIME_SET_CURRENT(t1);
	if (phase_s)
		phase_s[7] = INSTR_TIME_GET_DOUBLE(t1) - INSTR_TIME_GET_DOUBLE(t0);	/* setup: the counting scan + DSM */

	PG_TRY();
	{
		tp_participate(sh, area, heap, index, indexInfo, phase_s);
	}
	PG_CATCH();
	{
		pg_atomic_write_u32(&sh->failed, 1);
		PG_RE_THROW();
	}
	PG_END_TRY();
	BarrierDetach(&sh->barrier);
	if (pcxt)
		WaitForParallelWorkersToFinish(pcxt);

	/* ---- write the pages: TIDs (id order) then the image, straight from the area ---- */
	if (sh->n >= 1)
	{
		FlatHeader *img = (FlatHeader *) (area + sh->img_off);
		BlockNumber start = InvalidBlockNumber;

		meta->n = sh->n;
		meta->C = sh->C;
		meta->core_bytes = img->total_bytes;
		meta->blob_bytes = (uint64) sh->n * sizeof(uint64) + img->total_bytes;
		meta->var_captured = sh->var;
		for (l = 0; l < L; l++)
			meta->ch_default[l] = img->ch[l];
		{
			const uint64 *tids = A(area, sh->tids_off, uint64);
			const int  *order = A(area, sh->order_off, int);
			uint64	   *sorted = (uint64 *) palloc(sizeof(uint64) * (Size) sh->n);
			int			i;

			for (i = 0; i < sh->n; i++)
				sorted[i] = tids[order[i]];	/* the TID table in id order */
			INSTR_TIME_SET_CURRENT(t0);
			tp_write_index2(index, MAIN_FORKNUM, meta, (const char *) sorted, (uint64) sh->n * sizeof(uint64),
							(const char *) img, img->total_bytes, &start);
			pfree(sorted);
		}
		INSTR_TIME_SET_CURRENT(t1);
		if (phase_s)
			phase_s[8] = INSTR_TIME_GET_DOUBLE(t1) - INSTR_TIME_GET_DOUBLE(t0);
	}
	{
		int64		n = sh->n;

		if (IsMVCCSnapshot(snapshot))
			UnregisterSnapshot(snapshot);
		if (pcxt)
		{
			DestroyParallelContext(pcxt);
			ExitParallelMode();
		}
		else
			dsm_detach(ownseg);
		pfree(leader);
		return n;
	}
}
