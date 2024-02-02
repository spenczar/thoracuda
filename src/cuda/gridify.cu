
#include <stdlib.h>
#include <cstdlib>
#include "cuda_macros.h"
#include "pair.h"
#include "pairminmax.h"

#define GRID_WIDTH_CELLS 64
#define GRID_HEIGHT_CELLS 64

struct IndexPair {
  int start;
  int end;
};

struct Grid {
  struct IndexPair* address_map;

  int n;
  struct XYPair *xys;
  double *ts;
};

struct CellPosition {
  short x;
  short y;
  short t;
};

struct CellPosition xy_to_cell(struct XYPair xy, struct XYBounds bounds, double t, double t_min) {
  return {
    .x = (short)((xy.x - bounds.xmin) / (bounds.xmax - bounds.xmin) * GRID_WIDTH_CELLS),
    .y = (short)((xy.y - bounds.ymin) / (bounds.ymax - bounds.ymin) * GRID_HEIGHT_CELLS),
    .t = (short)(t - t_min)
  };
};

struct SortableData {
  struct XYPair *xys;
  double *ts;

  struct XYBounds bounds;
  double t_min;
};

int xy_compare_txy(const void *a, const void *b, void *data) {
  int ai = *(int*)a;
  int bi = *(int*)b;
  struct SortableData *sd = (struct SortableData*)data;
  struct XYPair *xys = sd->xys;
  double *ts = sd->ts;

  struct CellPosition a_cp = xy_to_cell(xys[ai], sd->bounds, ts[ai], sd->t_min);
  struct CellPosition b_cp = xy_to_cell(xys[bi], sd->bounds, ts[bi], sd->t_min);

  if (a_cp.t < b_cp.t) {
    return -1;
  } else if (a_cp.t > b_cp.t) {
    return 1;
  } else if (a_cp.x < b_cp.x) {
    return -1;
  } else if (a_cp.x > b_cp.x) {
    return 1;
  } else if (a_cp.y < b_cp.y) {
    return -1;
  } else if (a_cp.y > b_cp.y) {
    return 1;
  } else {
    return 0;
  }
}

double min_d(double *d, int n) {
  double min = d[0];
  for (int i = 1; i < n; i++) {
    if (d[i] < min) {
      min = d[i];
    }
  }
  return min;
}

int gridify_points_serial(struct XYPair* xys, double *t, int n, struct Grid *grid) {
  // First, compute min and max of x and y. This sets the bounds of the grid.
  struct XYBounds bounds = xy_bounds_serial(xys, n);
  struct XYPair *xys_reindexed = NULL;
  double *ts_reindexed = NULL;
  struct SortableData sd;  
  int address_map_size;
  
  int *indexes = (int*)malloc(sizeof(int) * n);
  if (!indexes) {
    goto fail;
  }

  for (int i = 0; i < n; i++) {
    indexes[i] = i;
  }

  // Sort all points by t, then x, then y.
  sd = {
    .xys = xys,
    .ts = t,
    .bounds = bounds,
    .t_min = min_d(t, n)
  };
  qsort_r(indexes, n, sizeof(int), xy_compare_txy, &sd);

  // Reindex points and ts.
  xys_reindexed = (struct XYPair*)malloc(sizeof(struct XYPair) * n);
  if (!xys_reindexed) {
    goto fail;
  }
  ts_reindexed = (double*)malloc(sizeof(double) * n);
  if (!ts_reindexed) {
    goto fail;
  }
  for (int i = 0; i < n; i++) {
    xys_reindexed[i] = xys[indexes[i]];
    ts_reindexed[i] = t[indexes[i]];
  }

  grid->n = n;
  grid->xys = xys_reindexed;
  grid->ts = ts_reindexed;
  
  // Now, we can start to gridify the points.
  address_map_size = GRID_WIDTH_CELLS * GRID_HEIGHT_CELLS * (ts_reindexed[n - 1] - ts_reindexed[0] + 1);
  grid->address_map = (struct IndexPair*)malloc(sizeof(struct IndexPair) * address_map_size);
  if (!grid->address_map) {
    goto fail;
  }
  // Initialize to -1
  for (int i = 0; i < address_map_size; i++) {
    grid->address_map[i].start = -1;
    grid->address_map[i].end = -1;
  }
  
  for (int i = 0; i < n; i++) {
    struct CellPosition cp = xy_to_cell(xys_reindexed[i], bounds, ts_reindexed[i], sd.t_min);
    int index = cp.t * GRID_WIDTH_CELLS * GRID_HEIGHT_CELLS + cp.y * GRID_WIDTH_CELLS + cp.x;
    if (grid->address_map[index].start == -1) {
      grid->address_map[index].start = i;
    }
    grid->address_map[index].end = i;
  }

 ok:
  if (indexes) {
    free(indexes);
  }
  return 0;
  
 fail:
  if (ts_reindexed) {
    free(ts_reindexed);
  }
  if (xys_reindexed) {
    free(xys_reindexed);
  }
  if (indexes) {
    free(indexes);
  }
  return 1;
}

