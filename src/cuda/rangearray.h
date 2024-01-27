#pragma once

#include <vector>
#include <utility>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

class Exposure {
 public:
  // invariant: x.size() == y.size()
  std::vector<float> x;
  std::vector<float> y;
  std::vector<int> id;
  double t; // MJD

  Exposure(std::vector<float> x, std::vector<float> y, std::vector<int> id, double t) :
  x(x), y(y), id(id), t(t) {}
};


struct FindPairConfig {
  float max_vx;
  float max_vy;

  float min_x;
  float max_x;
  
  float min_y;
  float max_y;

  int grid_dim_x;
  int grid_dim_y;
};


class DeviceGrid {
public:
  thrust::device_vector<int2> grid;
  thrust::device_vector<int> point_ids;
  thrust::device_vector<float2> points;
};

class GridCoordKernel : thrust::unary_function<float2, short2> {
  int min_x, max_x, grid_dim_x;
  int min_y, max_y, grid_dim_y;
public:
  GridCoordKernel(int min_x, int max_x, int grid_dim_x,
		  int min_y, int max_y, int grid_dim_y);
  __host__ __device__ short2 operator()(float2 xy);
};

DeviceGrid build_grid(const Exposure& e, const FindPairConfig& config);


thrust::device_vector<short2> build_grid_coord_map(const thrust::device_vector<float2>& xy_d,
						   const FindPairConfig& config);

void sort_by_grid_cell(thrust::device_vector<short2>& grid_coords,
		       thrust::device_vector<float2>& xy,
		       thrust::device_vector<int>& ids);


thrust::device_vector<int2> build_dense_matrix(thrust::device_vector<short2>& grid_coords,
					       int grid_dim_y);

__global__ void build_dense_matrix_kernel(short2 *grid_coords_data,
					  int n,
					  int grid_dim_y,
					  int2 *matrix);
