#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <utility>
#include <vector>

class Exposure {
 public:
  // invariant: x.size() == y.size()
  std::vector<float> x;
  std::vector<float> y;
  std::vector<int> id;
  double t;  // MJD

  Exposure(std::vector<float> x, std::vector<float> y, std::vector<int> id, double t) : x(x), y(y), id(id), t(t) {}
};

struct FindPairConfig {
  float max_vx;
  float max_vy;

  float min_x;
  float max_x;

  float min_y;
  float max_y;

  int grid_n_cells_1d;
};

/// GridQuery represents a query to a DeviceGrid. It represents the
/// collection of circles that are going to be queried against the
/// grid.
struct GridQuery {
  // Invariant: point_ids.size() == points.size()

  /// IDs of points in the query
  std::vector<int> point_ids;

  /// Points in the query. The points are in the same order as the
  /// point_ids.
  std::vector<float2> points;

  /// Time, in MJD, of the points.
  double t;

  /// The maximum velocity of the points in the query. This is used to
  /// determine the maximum distance a point can travel in the time
  /// interval between the query time and the target time.
  float max_vx;
  float max_vy;
};

/// DeviceGrid represents a collection of point source detections
/// which have been aligned to a uniform grid, stored on the GPU
/// (hence "Device").
class DeviceGrid {
 public:
  thrust::device_vector<int2> grid;
  thrust::device_vector<int> point_ids;
  thrust::device_vector<float2> points;
};

/// Kernel that computes the grid coordinates of a point.
class GridCoordKernel : thrust::unary_function<float2, short2> {
  /// The minimum and maximum x and y values of the overall collection
  /// of points.
  float min_x, max_x, min_y, max_y;
  /// The number of cells in the grid. The grid is square.
  int grid_n_cells_1d;
  /// The width of a grid cell. Note that they are always square.
  float grid_cell_width;

 public:
  GridCoordKernel(float min_x, float max_x, float min_y, float max_y, int grid_n_cells_1d);
  /// Compute the grid coordinates of a point.
  __host__ __device__ short2 operator()(float2 xy);
};

DeviceGrid build_grid(const Exposure& e, const FindPairConfig& config);

thrust::device_vector<short2> build_grid_coord_map(const thrust::device_vector<float2>& xy_d,
                                                   const FindPairConfig& config);

void sort_by_grid_cell(thrust::device_vector<short2>& grid_coords, thrust::device_vector<float2>& xy,
                       thrust::device_vector<int>& ids);

thrust::device_vector<int2> build_dense_matrix(thrust::device_vector<short2>& grid_coords, int grid_dim_y);

__global__ void build_dense_matrix_kernel(short2* grid_coords_data, int n, int grid_dim_y, int2* matrix);
