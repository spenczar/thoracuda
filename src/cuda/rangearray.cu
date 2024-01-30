/*
 * The idea here is to build a data structure which facilities quick
 * "range queries" in two dimensions using CUDA.
 *
 *
 * The input data is [x, y] pairs. Both x and y are floats, and are
 * confined to a relatively narrow domain - a fixed range from -10 to
 * +10 would be typical, and represents a 10-degree radius with
 * respect to the test orbit. Ideally, the code should be configurable
 * with respect to this range, but this requirement may be dropped in
 * early editions of this code.
 *
 * The queries will be for relatively narrow ranges, like a radius of
 * +/- 0.1 around a target point. The radius of a query (which will
 * always be for a circular region) will be determined by the maximum
 * tolerated velocity offset between detection and test orbit, as well
 * as the maximum analyzed time difference between two exposures.
 *
 * Typical maximum velocity is +/- 0.1 vx, +/- 0.1 vy, both per
 * day. Typical maximum day offset is 10 days. so +/- 1 degree is
 * possible.
 *
 * The data structure is built around a grid overlaid on the x-y
 * points. The grid's resolution isn't certain yet - it probably has a
 * lot of tradeoffs. The data structure, one constructed, has two
 * parts:
 *
 *   1. A contiguous array of the x-y points, sorted so that all x-y
 *      points in the same grid cell appear sequentially.
 *
 *   2. A dense matrix of int32 pairs, M. M[i, j] corresponds to
 *      asking about grid cell i,j, and contains the offset into the
 *      array where its points can be found, as well as the offset to
 *      the _end_ of the sequence of points. If the cell has _no_
 *      points, then the cell will hold (0,0).
 *
 *   3. A contiguous array of all point IDs, in same order as 1.
 *
 * The dense matrix can be queried using texture APIs to make
 * efficient use of the GPU'ss cache. These ensure that the cache for
 * an SMP will arrange datapoints in a way that improves cache
 * locality in two dimensions.
 *
 * A range query for points between [xmin, xmax] and [ymin, ymax] is
 * performed by:
 *
 * 0. For an input point ID, and [xmin, xmax, ymin, ymax]:
 *
 * 1. calculate all the [i, j] pairs within the [xmin, xmax] and
 *    [ymin, ymax] bounds.
 *
 * 2. For each [i, j] pair, a thread checks M, and if the count is >
 *    0, it writes <count> integers (from <start_pos> to <end_pos>)
 *    into some sort of shared array for input point.
 *
 * 3. That shared array is condensed and sorted. Now it contains
 *    indexes that could be queried in parallel, to inspect and see if
 *    each point is withiin range ofthe original point.
 *
 * 4. For each such point, a threadd can emit the input point ID and
 *    matching point ID. The matching point ID is retrieved from array
 *    3
 */

#include <thrust/sort.h>

#include "rangearray.h"

thrust::host_vector<float2> exposure_xy_data(const Exposure& e) {
  thrust::host_vector<float2> result(e.x.size());
  for (int i = 0; i < e.x.size(); ++i) {
    result[i] = make_float2(e.x[i], e.y[i]);
  }
  return result;
}

// Auxiliary for sorting grid coords
struct short2_compare {
  __host__ __device__ bool operator()(const short2& a, const short2& b) {
    if (a.x < b.x) {
      return true;
    } else if (a.x > b.x) {
      return false;
    } else {
      return a.y < b.y;
    }
  }
};

void sort_by_grid_cell(thrust::device_vector<short2>& grid_coords, thrust::device_vector<float2>& xy,
                       thrust::device_vector<int>& ids) {
  auto zipped = thrust::make_zip_iterator(thrust::make_tuple(xy.begin(), ids.begin()));
  thrust::sort_by_key(grid_coords.begin(), grid_coords.end(), zipped, short2_compare());
}

DeviceGrid build_grid(const Exposure& e, const FindPairConfig& config) {
  DeviceGrid result;

  // Load x, y, and IDs into device memory.
  thrust::host_vector<float2> xy_h = exposure_xy_data(e);
  thrust::device_vector<float2> xy_d(xy_h);

  thrust::host_vector<int> ids_h(e.id);
  thrust::device_vector<int> ids_d(ids_h);

  // Step 1: map x and y to grid cells.
  thrust::device_vector<short2> grid_coords = build_grid_coord_map(xy_d, config);

  // Step 2: sort by grid cell.
  sort_by_grid_cell(grid_coords, xy_d, ids_d);

  // Step 3: build the dense matrixes.
  result.grid = build_dense_matrix(grid_coords, config.grid_n_cells_1d);
  result.point_ids = ids_d;
  result.points = xy_d;
  return result;
}

thrust::device_vector<short2> build_grid_coord_map(const thrust::device_vector<float2>& xy_d,
                                                   const FindPairConfig& config) {
  // map x and y to grid cells. Grid cells are notated as a
  // pair of ints, [i, j]. i is the x coordinate, j is the y
  // coordinate. The grid is a fixed size, and is configured by the
  // FindPairConfig.
  thrust::device_vector<short2> grid_coords(xy_d.size());
  thrust::transform(xy_d.begin(), xy_d.end(), grid_coords.begin(),
                    GridCoordKernel(config.min_x, config.max_x, config.min_y, config.max_y, config.grid_n_cells_1d));
  return grid_coords;
}

GridCoordKernel::GridCoordKernel(float min_x, float max_x, float min_y, float max_y, int grid_n_cells_1d)
    : min_x(min_x), max_x(max_x), min_y(min_y), max_y(max_y), grid_n_cells_1d(grid_n_cells_1d) {
  float x_extent = max_x - min_x;
  float y_extent = max_y - min_y;
  if (x_extent > y_extent) {
    // x is the limiting dimension
    grid_cell_width = x_extent / grid_n_cells_1d;
  } else {
    // y is the limiting dimension
    grid_cell_width = y_extent / grid_n_cells_1d;
  }
}

short2 GridCoordKernel::operator()(float2 xy) {
  short2 result;
  result.x = (xy.x - min_x) / grid_cell_width;
  result.y = (xy.y - min_y) / grid_cell_width;
  return result;
}

thrust::device_vector<int2> build_dense_matrix(thrust::device_vector<short2>& grid_coords, int grid_dim_y) {
  thrust::device_vector<int2> matrix(grid_dim_y * grid_dim_y);
  int n = grid_coords.size();
  int block_size = 256;
  int n_blocks = (n + block_size - 1) / block_size;
  build_dense_matrix_kernel<<<n_blocks, block_size>>>(thrust::raw_pointer_cast(grid_coords.data()), n, grid_dim_y,
                                                      thrust::raw_pointer_cast(matrix.data()));
  return matrix;
}
__global__ void build_dense_matrix_kernel(short2* grid_coords_data, int n, int grid_dim_y, int2* matrix) {
  /*
   * This kernel is responsible for building the dense matrix. It
   * requires that grid_coords_data be sorted by grid cell, and that
   * the matrix be initialized to all zeros.
   *
   * The (i, j)th cell of the matrix will be populated if there is at
   * least one value of (i, j) in grid_coords_data.
   *
   * The cell will have a pair of values indicating the position of
   * the sequence of (i, j) values. The first value in the pair is the
   * index to the start of the sequence, and the second value is the
   * (exclusive) index for the end of the sequence.
   */
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n) {
    return;
  }
  short2 grid_coord = grid_coords_data[i];

  int start_offset = -1;
  int end_offset = -1;
  if (i == 0) {
    // Special case: first cell can't handle looking backwards. It is
    // always a start.
    start_offset = 0;
  } else {
    short2 prev_grid_coord = grid_coords_data[i - 1];
    if ((grid_coord.x != prev_grid_coord.x) || (grid_coord.y != prev_grid_coord.y)) {
      // present index is a start offset for the new grid coord
      start_offset = i;
    }
  }

  if (i == (n - 1)) {
    end_offset = n;
  } else {
    short2 next_grid_coord = grid_coords_data[i + 1];
    if ((grid_coord.x != next_grid_coord.x) || (grid_coord.y != next_grid_coord.y)) {
      // present index is an end offset for the new grid coord
      end_offset = i + 1;
    }
  }

  if (start_offset < 0 && end_offset < 0) {
    // No change to the matrix, since we're in the middle of a run.
    return;
  }

  int dense_idx = grid_coord.x * grid_dim_y + grid_coord.y;
  if (start_offset >= 0) {
    matrix[dense_idx].x = start_offset;
  }
  if (end_offset >= 0) {
    matrix[dense_idx].y = end_offset;
  }
}