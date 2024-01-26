#pragma once

#include <vector>
#include <utility>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

typedef std::pair<int, int> int_pair;

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
  thrust::device_vector<int> grid_offsets;
  thrust::device_vector<int> grid_counts;
  thrust::device_vector<int> point_ids;
  thrust::device_vector<float2> points;
};

struct IntPair {
  int x;
  int y;
};

class GridCoordKernel : thrust::binary_function<float, float, int_pair> {
  int min_x, max_x, grid_dim_x;
  int min_y, max_y, grid_dim_y;
public:
  GridCoordKernel(int min_x, int max_x, int grid_dim_x,
		  int min_y, int max_y, int grid_dim_y);
  __host__ __device__ struct IntPair operator()(float x, float y);
};

DeviceGrid build_grid(const Exposure& e, const FindPairConfig& config);
thrust::device_vector<struct IntPair> build_grid_coord_map(const thrust::device_vector<float>& x_d,
							   const thrust::device_vector<float>& y_d,
							   const FindPairConfig& config);


/* Exposed for testing */
thrust::host_vector<struct IntPair> build_grid_coord_map(const thrust::host_vector<float>& x_d,
							 const thrust::host_vector<float>& y_d,
							 const FindPairConfig& config);
