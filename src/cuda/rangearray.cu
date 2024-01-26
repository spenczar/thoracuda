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
 *      array where its points can be found, as well as the number of
 *      points in that cell. If the cell has _no_ points, then the
 *      offset will be negative.
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
 *    0, it writes <count> integers (from <start_pos> to <start_pos +
 *    count>) into some sort of shared array for input point.
 *
 * 3. That shared array is condensed and sorted. Now it contains
 *    indexes that could be queried in parallel, to inspect and see if
 *    each point is withiin range ofthe original point.
 *
 * 4. For each such point, a threadd can emit the input point ID and
 *    matching point ID. The matching point ID is retrieved from array
 *    3
 */

#include "rangearray.h"

DeviceGrid build_grid(const Exposure& e,
		      const FindPairConfig& config) {
  DeviceGrid result;

  // Load x and y into device memory.
  thrust::host_vector<float> x(e.x);
  thrust::host_vector<float> y(e.y);
  thrust::device_vector<float> x_d(x.size());
  thrust::device_vector<float> y_d(y.size());
  thrust::copy(x.begin(), x.end(), x_d.begin());
  thrust::copy(y.begin(), y.end(), y_d.begin());
  
  // Step 1: map x and y to grid cells.
  thrust::device_vector<struct IntPair> grid_coords =
    build_grid_coord_map(x_d, y_d, config);

  // Step 2: sort the grid coords by grid cell.
  return result;
}

thrust::device_vector<struct IntPair> build_grid_coord_map(const thrust::device_vector<float>& x_d,
							   const thrust::device_vector<float>& y_d,
							   const FindPairConfig& config) {
  // map x and y to grid cells. Grid cells are notated as a
  // pair of ints, [i, j]. i is the x coordinate, j is the y
  // coordinate. The grid is a fixed size, and is configured by the
  // FindPairConfig.
  thrust::device_vector<struct IntPair> grid_coords(x_d.size());
  thrust::transform(x_d.begin(), x_d.end(), y_d.begin(),
		    grid_coords.begin(),
		    GridCoordKernel(config.min_x, config.max_x, config.grid_dim_x,
				    config.min_y, config.max_y, config.grid_dim_y));
  return grid_coords;
}

GridCoordKernel::GridCoordKernel(int min_x, int max_x, int grid_dim_x,
				 int min_y, int max_y, int grid_dim_y) :
  min_x(min_x), max_x(max_x), grid_dim_x(grid_dim_x),
  min_y(min_y), max_y(max_y), grid_dim_y(grid_dim_y) {}

struct IntPair GridCoordKernel::operator()(float x, float y) {
  struct IntPair result;
  result.x = (x - min_x) / (max_x - min_x) * grid_dim_x;
  result.y = (y - min_y) / (max_y - min_y) * grid_dim_y;
  return result;
}


/* Exposed for testing */
thrust::host_vector<struct IntPair> build_grid_coord_map(const thrust::host_vector<float>& x,
							 const thrust::host_vector<float>& y,
							 const FindPairConfig& config) {
  thrust::device_vector<float> x_d(x.size());
  thrust::device_vector<float> y_d(y.size());
  thrust::copy(x.begin(), x.end(), x_d.begin());
  thrust::copy(y.begin(), y.end(), y_d.begin());
  return build_grid_coord_map(x_d, y_d, config);
}
   