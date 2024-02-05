
#include <stdio.h>
#include <stdlib.h>

#include <cstdlib>

#include "cuda_macros.h"
#include "gridify.h"
#include "pair.h"
#include "pairminmax.h"

#define GRID_WIDTH_CELLS 64
#define GRID_HEIGHT_CELLS 64

static double min_d(double *d, int n) {
  double min = d[0];
  for (int i = 1; i < n; i++) {
    if (d[i] < min) {
      min = d[i];
    }
  }
  return min;
}

static int n_unique_t_rounded(double *t, int n) {
  // Count unique t, rounded to the nearest integer.
  int n_unique = 1;
  double last_t = t[0];
  for (int i = 1; i < n; i++) {
    if ((int)t[i] != (int)last_t) {
      n_unique++;
      last_t = t[i];
    }
  }
  return n_unique;
}

static int calc_address_map_size(int n_t) { return GRID_WIDTH_CELLS * GRID_HEIGHT_CELLS * n_t; }

__host__ __device__ struct CellPosition xy_to_cell(struct XYPair xy, struct XYBounds bounds, double t, double t_min) {
  return {.x = (short)((xy.x - bounds.xmin) / (bounds.xmax - bounds.xmin) * GRID_WIDTH_CELLS),
          .y = (short)((xy.y - bounds.ymin) / (bounds.ymax - bounds.ymin) * GRID_HEIGHT_CELLS),
          .t = (short)(t - t_min)};
};

int xy_compare_txy(const void *a, const void *b, void *data) {
  int ai = *(int *)a;
  int bi = *(int *)b;
  struct SortableData *sd = (struct SortableData *)data;
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

int gridify_points_serial(struct XYPair *xys, double *t, int n, struct Grid *grid) {
  // First, compute min and max of x and y. This sets the bounds of the grid.
  struct XYBounds bounds = xy_bounds_serial(xys, n);
  struct XYPair *xys_reindexed = NULL;
  double *ts_reindexed = NULL;
  struct SortableData sd;
  int address_map_size = calc_address_map_size(n_unique_t_rounded(t, n));

  int *indexes = (int *)malloc(sizeof(int) * n);
  if (!indexes) {
    goto fail;
  }

  for (int i = 0; i < n; i++) {
    indexes[i] = i;
  }

  // Sort all points by t, then x, then y.
  sd = {.xys = xys, .ts = t, .bounds = bounds, .t_min = min_d(t, n)};
  qsort_r(indexes, n, sizeof(int), xy_compare_txy, &sd);

  // Reindex points and ts.
  xys_reindexed = (struct XYPair *)malloc(sizeof(struct XYPair) * n);
  if (!xys_reindexed) {
    goto fail;
  }
  ts_reindexed = (double *)malloc(sizeof(double) * n);
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
  address_map_size = GRID_WIDTH_CELLS * GRID_HEIGHT_CELLS * n_unique_t_rounded(t, n);
  grid->address_map_n = address_map_size;
  grid->address_map = (struct IndexPair *)malloc(sizeof(struct IndexPair) * address_map_size);
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

int gridify_points_parallel(struct XYPair *xys, double *t, int n, struct Grid *grid) {
  cudaError_t err;
  // First, compute min and max of x and y. This sets the bounds of
  // the grid, which allows us to compute cell positions.
  struct XYBounds bounds = xy_bounds_serial(xys, n);
  double t_min = min_d(t, n);
  int n_t = n_unique_t_rounded(t, n);
  int address_map_size = calc_address_map_size(n_t);

  struct XYPair *d_xys = NULL;
  double *d_ts = NULL;
  struct Grid *d_grid = NULL;

  // Copy data to the device
  CUDA_OR_FAIL(cudaMalloc((void **)&d_xys, sizeof(struct XYPair) * n));
  CUDA_OR_FAIL(cudaMalloc((void **)&d_ts, sizeof(double) * n));
  CUDA_OR_FAIL(cudaMemcpy(d_xys, xys, sizeof(struct XYPair) * n, cudaMemcpyHostToDevice));
  CUDA_OR_FAIL(cudaMemcpy(d_ts, t, sizeof(double) * n, cudaMemcpyHostToDevice));

  // Allocate space for the grid on the device
  CUDA_OR_FAIL(cudaMalloc((void **)&d_grid, sizeof(struct Grid)));
  CUDA_OR_FAIL(cudaMalloc((void **)&d_grid->xys, sizeof(struct XYPair) * n));
  CUDA_OR_FAIL(cudaMalloc((void **)&d_grid->ts, sizeof(double) * n));
  CUDA_OR_FAIL(cudaMalloc((void **)&d_grid->address_map, sizeof(struct IndexPair) * address_map_size));

  // Now, we can start to gridify the points.
  CUDA_OR_FAIL(gridify_points_parallel_on_device(d_xys, d_ts, n, d_grid, bounds, t_min));

  // Copy the grid back to the host
  grid->n = n;
  CUDA_OR_FAIL(cudaMemcpy(grid->xys, d_grid->xys, sizeof(struct XYPair) * n, cudaMemcpyDeviceToHost));
  CUDA_OR_FAIL(cudaMemcpy(grid->ts, d_grid->ts, sizeof(double) * n, cudaMemcpyDeviceToHost));
  CUDA_OR_FAIL(cudaMemcpy(grid->address_map, d_grid->address_map, sizeof(struct IndexPair) * address_map_size,
                          cudaMemcpyDeviceToHost));

  // Free device memory
  cudaFree(d_xys);
  cudaFree(d_ts);
  cudaFree(d_grid);
  return 0;

fail:
  if (d_xys) {
    cudaFree(d_xys);
  }
  if (d_ts) {
    cudaFree(d_ts);
  }
  if (d_grid) {
    cudaFree(d_grid);
  }
  return 1;
}

__host__ cudaError_t gridify_points_parallel_on_device(struct XYPair *d_xys, double *d_t, int n, struct Grid *d_grid,
                                                       struct XYBounds bounds, double t_min) {
  // Parallel components, which expect data to already be on the
  // device.
  // 1. Map xys to cell positions (x, y, t).
  //
  // 2. Sort the cell positions to be in order of t, then x, then
  //    y. Use radix sort. Simultaneously sort the original xys and t
  //    arrays.
  //
  // 3. Build the address map grid by "reducing" the cell position
  //    array.
  cudaError_t err;
  struct CellPosition *d_cell_positions = NULL;
  int threads_per_block = 256;
  int n_blocks = (n + threads_per_block - 1) / threads_per_block;

  CUDA_OR_FAIL(cudaMalloc((void **)&d_cell_positions, sizeof(struct CellPosition) * n));
  map_points_to_cells_kernel<<<n_blocks, threads_per_block>>>(d_xys, d_t, n, bounds, t_min, d_cell_positions);

ok:
  if (d_cell_positions) {
    cudaFree(d_cell_positions);
  }
  return cudaSuccess;
fail:
  if (d_cell_positions) {
    cudaFree(d_cell_positions);
  }
  return err;
}

__global__ void map_points_to_cells_kernel(struct XYPair *xys, double *t, int n, struct XYBounds bounds, double t_min,
                                           struct CellPosition *cell_positions) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > n) {
    return;
  }
  cell_positions[i] = xy_to_cell(xys[i], bounds, t[i], t_min);
}

struct IndexPair grid_query(struct CellPosition cp, struct Grid *grid) {
  int index = cp.t * GRID_WIDTH_CELLS * GRID_HEIGHT_CELLS + cp.y * GRID_WIDTH_CELLS + cp.x;
  return grid->address_map[index];
}