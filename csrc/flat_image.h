/* The flat (frozen) index image: the extension's shared segment and, since E2, its page format.
 * Position-independent — a header of byte offsets from its own start, every array 64-byte aligned —
 * so several processes map one copy and search it in place. Written by the core (cascade_flatten /
 * cascade_reflatten) or assembled in place by the extension's parallel build (cascade_flat_layout gives
 * the offsets; the participants fill the arrays). Plain C so pgext/ can address the arrays directly.
 *
 *   lo, step        float [d3]            quantisation of the PCA head into int8
 *   proj            float [d3][D]         PCA rows (orthonormal), mean float [D]
 *   cent            float [C][d3]         cell centroids in the head space
 *   cal[l]          float [cal_cap[l]]    sorted critical cosines of the calibration pairs (cal_n[l] stored)
 *   perm            int32 [n]             position -> id;  pos_of int32 [npos_of]: id -> position (-1 unused)
 *   cell_of         int32 [n]             position -> cell;  deleted uint8 [n]: tombstones at image time
 *   err[l], tail[l] float [n]             per level: quantisation error norm, PCA tail norm (by position)
 *   blk_code_off    uint64 [C + 1]        byte offsets of each cell's level-blocked codes inside blk_codes
 *   blk_pos_off     uint64 [C + 1]        element offsets of each cell's positions inside blk_pos (int32)
 *   graph           FlatGraph blob        the router (flat_graph.h): codes, level-0 lists, upper layers
 */
#ifndef TRIGPOINT_FLAT_IMAGE_H
#define TRIGPOINT_FLAT_IMAGE_H

#include <stdint.h>

#define TPFL_MAGIC 0x4c465054u	/* "TPFL" */
#define TPFL_VERSION 2u

typedef struct FlatHeader
{
	uint32_t	magic,
				version;
	int32_t		n,
				D,
				C,
				L,
				lv[4];
	int32_t		M,
				efC,
				npos_of,
				router_mode;
	long long	n_inserted,
				n_deleted,
				n_compacted;
	float		ch[4],
				var_captured;
	uint64_t	total_bytes;
	uint64_t	cal_n[4],
				cal_cap[4];
	uint64_t	lo,
				step,
				proj,
				mean,
				cent,
				cal[4];
	uint64_t	perm,
				pos_of,
				cell_of,
				deleted;
	uint64_t	err[4],
				tail[4];
	uint64_t	blk_code_off,
				blk_pos_off,
				blk_codes,
				blk_pos;
	uint64_t	graph,
				graph_bytes;
} FlatHeader;

#define FLAT_PTR(h, off, T) ((T *) ((char *) (h) + (off)))
#define FLAT_CPTR(h, off, T) ((const T *) ((const char *) (h) + (off)))

#endif
