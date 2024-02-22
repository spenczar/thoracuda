#pragma once

#include <vector>

#include "clustering/inputs.cuh"

using thoracuda::clustering::Inputs;

namespace thoracuda {
namespace clustering {
  std::vector<int> DBSCAN(float epsilon, int min_points, const Inputs &inputs);
  __global__ void identify_core_points(int *counts, int n, int min_points, int *cluster_ids);
  __global__ void collapse_clusters(int *cluster_ids, int n, int *offsets, int *counts, int *neighbors);
  __global__ void associate_points_to_core_points(int *counts, int n, int *offsets, int *neighbors, int *cluster_ids, int min_points);
  __device__ int min_neighbor_cluster_id(int i, int *cluster_ids, int *offsets, int *counts, int *neighbors);      
}
}