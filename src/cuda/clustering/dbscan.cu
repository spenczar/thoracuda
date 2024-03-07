
#include <vector>

#include "clustering/dbscan.cuh"
#include "clustering/inputs.cuh"
#include "cuda_macros.h"
#include "rangequery/counts_table.cuh"
#include "rangequery/data_handle.cuh"
#include "rangequery/neighbors_table.cuh"
#include "rangequery/offsets_table.cuh"

using thoracuda::clustering::Inputs;
using thoracuda::rangequery::CountsTable;
using thoracuda::rangequery::DataHandle;
using thoracuda::rangequery::NeighborsTable;
using thoracuda::rangequery::OffsetsTable;

namespace thoracuda {
namespace clustering {
std::vector<int> DBSCAN(float epsilon, int min_points, const Inputs &inputs) {
  cudaError_t err;

  // 1. Count points within epsilon.
  DataHandle data(inputs.x, inputs.y, inputs.t, inputs.id);

  CountsTable counts(data, epsilon);

  // 2. Identify core points
  int *cluster_ids;
  CUDA_OR_THROW(cudaMalloc(&cluster_ids, data.n * sizeof(int)));
  int threads_per_block = 256;
  int blocks = (data.n + threads_per_block - 1) / threads_per_block;
  identify_core_points<<<blocks, threads_per_block>>>(counts.counts, counts.n_counts, min_points, cluster_ids);

  // 3. Merge core points into clusters
  OffsetsTable offsets(counts);
  NeighborsTable neighbors(data, epsilon, offsets);
  collapse_clusters<<<blocks, threads_per_block>>>(cluster_ids, data.n, offsets.offsets, counts.counts,
                                                   neighbors.neighbors);

  // 4. Associate border points to clusters
  associate_points_to_core_points<<<blocks, threads_per_block>>>(counts.counts, counts.n_counts, offsets.offsets,
                                                                 neighbors.neighbors, cluster_ids, min_points);

  std::vector<int> host_cluster_ids(data.n);
  CUDA_OR_THROW(cudaMemcpy(host_cluster_ids.data(), cluster_ids, data.n * sizeof(int), cudaMemcpyDeviceToHost));

  cudaFree(cluster_ids);

  return host_cluster_ids;
}

__global__ void identify_core_points(int *counts, int n, int min_points, int *cluster_ids) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for (; i < n; i += blockDim.x * gridDim.x) {
    if (counts[i] >= (min_points - 1)) {
      cluster_ids[i] = i;
    } else {
      cluster_ids[i] = -1;
    }
  }
}

__global__ void associate_points_to_core_points(int *counts, int n, int *offsets, int *neighbors, int *cluster_ids,
                                                int min_points) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for (; i < n; i += blockDim.x * gridDim.x) {
    if (counts[i] > 0 && counts[i] < (min_points - 1)) {
      int start = i == 0 ? 0 : offsets[i - 1];
      int end = offsets[i];
      int cluster_id = -1;
      for (int j = start; j < end; j++) {
        int neighbor_id = neighbors[j];
        if (cluster_ids[neighbor_id] != -1) {
          cluster_id = cluster_ids[neighbor_id];
          break;
        }
      }
      cluster_ids[i] = cluster_id;
    }
  }
}

__global__ void collapse_clusters(int *cluster_ids, int n, int *offsets, int *counts, int *neighbors) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for (; i < n; i += blockDim.x * gridDim.x) {
    int cluster_id = cluster_ids[i];
    if (cluster_id == -1) {
      continue;
    }
    int ids_to_update[16];
    int n_ids_to_update = 0;

    // Recursively find the root of the cluster
    while (true) {
      int merge_into = min_neighbor_cluster_id(i, cluster_ids, offsets, counts, neighbors);
      if (merge_into == cluster_id) {
        // Found the root
        break;
      }
      if (n_ids_to_update < 16) {
        ids_to_update[n_ids_to_update] = i;
        n_ids_to_update++;
      }
      cluster_id = merge_into;
    }

    if (n_ids_to_update > 0) {
      for (int j = 0; j < n_ids_to_update; j++) {
        cluster_ids[ids_to_update[j]] = cluster_id;
      }
    }
  }
}

__device__ int min_neighbor_cluster_id(int i, int *cluster_ids, int *offsets, int *counts, int *neighbors) {
  int cluster_id = cluster_ids[i];
  if (cluster_id == -1) {
    return -1;
  }
  // Subtle: this indexing scheme is tightly linked to use of
  // exclusive vs inclusive scan
  int start = i == 0 ? 0 : offsets[i - 1];
  int end = offsets[i];
  for (int j = start; j < end; j++) {
    int neighbor_id = neighbors[j];
    int neighbor_cluster_id = cluster_ids[neighbor_id];
    if (neighbor_cluster_id != -1 && neighbor_cluster_id < cluster_id) {
      cluster_id = neighbor_cluster_id;
    }
  }
  return cluster_id;
}

}  // namespace clustering
}  // namespace thoracuda