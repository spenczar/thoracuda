#include <assert.h>
#include <cuda_runtime.h>

#include <cub/device/device_scan.cuh>

#include "cuda_macros.h"
#include "hotspot2d.h"
#include "pair.h"

cudaError_t copy_clusters_to_host(struct Cluster **clusters_d, int *n_clusters_d, struct Cluster **clusters_h,
                                  int *n_clusters_h) {
  cudaError_t err;
  CUDA_OR_FAIL(cudaMemcpy(n_clusters_h, n_clusters_d, sizeof(int), cudaMemcpyDeviceToHost));
  if (*n_clusters_h > 0) {
    *clusters_h = (struct Cluster *)malloc(*n_clusters_h * sizeof(struct Cluster));
    if (*clusters_h == NULL) {
      err = cudaErrorMemoryAllocation;
      goto fail;
    }
    CUDA_OR_FAIL(cudaMemcpy(*clusters_h, *clusters_d, *n_clusters_h * sizeof(struct Cluster), cudaMemcpyDeviceToHost));
    for (int i = 0; i < *n_clusters_h; i++) {
      (*clusters_h)[i].point_ids = (int *)malloc((*clusters_h)[i].n * sizeof(int));
      CUDA_OR_FAIL(cudaMemcpy((*clusters_h)[i].point_ids, (*clusters_d)[i].point_ids, (*clusters_h)[i].n * sizeof(int),
                              cudaMemcpyDeviceToHost));
    }
  }
fail:
  if (err != cudaSuccess) {
    if (*clusters_h) {
      for (int i = 0; i < *n_clusters_h; i++) {
        if ((*clusters_h)[i].point_ids) {
          free((*clusters_h)[i].point_ids);
        }
      }
      free(*clusters_h);
      *clusters_h = NULL;
    }
  }
  return err;
}

cudaError_t hotspot2d_parallel(const struct XYPair *xys, const float *ts, const int *ids, int n,
                               struct Hotspot2DParams params, struct Cluster **clusters, int *n_clusters) {
  cudaError_t err;
  struct XYPair *xys_d;
  float *ts_d;
  int *ids_d;
  struct Cluster **clusters_d;
  int *n_clusters_d;

  CUDA_OR_FAIL(cudaMalloc((void **)&xys_d, n * sizeof(struct XYPair)));
  CUDA_OR_FAIL(cudaMalloc((void **)&ts_d, n * sizeof(float)));
  CUDA_OR_FAIL(cudaMalloc((void **)&ids_d, n * sizeof(int)));
  CUDA_OR_FAIL(cudaMalloc((void **)&clusters_d, sizeof(struct Cluster *)));
  CUDA_OR_FAIL(cudaMalloc((void **)&n_clusters_d, sizeof(int)));

  CUDA_OR_FAIL(cudaMemcpy(xys_d, xys, n * sizeof(struct XYPair), cudaMemcpyHostToDevice));
  CUDA_OR_FAIL(cudaMemcpy(ts_d, ts, n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OR_FAIL(cudaMemcpy(ids_d, ids, n * sizeof(int), cudaMemcpyHostToDevice));

  err = _hotspot2d_parallel_on_device(xys_d, ts_d, ids_d, n, params, clusters_d, n_clusters_d);
  if (err != cudaSuccess) {
    goto fail;
  }

  err = copy_clusters_to_host(clusters_d, n_clusters_d, clusters, n_clusters);
  if (err != cudaSuccess) {
    goto fail;
  }

fail:
  if (xys_d) {
    cudaFree(xys_d);
  }
  if (ts_d) {
    cudaFree(ts_d);
  }
  if (ids_d) {
    cudaFree(ids_d);
  }
  if (clusters_d) {
    cudaFree(clusters_d);
  }
  if (n_clusters_d) {
    cudaFree(n_clusters_d);
  }
  return err;
}

cudaError_t _hotspot2d_parallel_on_device(const struct XYPair *xys_d, const float *ts_d, const int *ids_d, int n,
                                          struct Hotspot2DParams params, struct Cluster **clusters_d,
                                          int *n_clusters_d) {
  cudaError_t err;

  // First, find the "exposure boundaries:" the indexes when t values
  // change in ts_d.
  int *boundaries_d = NULL;
  int n_boundaries_d;
  CUDA_OR_FAIL(find_exposure_boundaries(ts_d, n, &boundaries_d, &n_boundaries_d));

  /*
   For each velocity offset (vx, vy) in the search space:

    A. Compute shifted xys based on the velocity offset and the time
       of the datapoint.
    B. Quantize the (shifted) xys to a uniform grid.
    C. Sort the quantized data. Simultaneously sort ids_d.
    D. Search the quantized data for runs of length > min_points.
       i. Compare element i with i - 1. Write a 1 if different, 0 if same.
       ii. Prefix sum the above.
       iii. Compact the indexes i using the prefix sums. The ith
         element of the result is the index that starts run i. Call this runs[].
       iv. Compute runs[j] - runs[j-1]. If > min_cluster_size, then the run
         starting at j and ending at j-1 in the sorted data is a cluster of
         interest. Emit the cluster length (runs[j] - runs[j-1]) if so, else 0.
         Call this lengths[].
       v. Prefix sum lengths[]. Call this length_sums[]. This
         becomes an array used to gather the clustered data
         points. The total tells us how much memory to allocate for
         in-cluster points.
       vi. Gather the clusters:
         for j in range(0, len(runs)):
           if lengths[j] > 0:
              data_start_idx = runs[j-1]
              data_end_idx = runs[j]
              output_start_idx = length_sums[j]
              k = 0
              while (data_start_idx < data_end_idx) {
                  out[output_start_idx + k] = ids[data_start_idx]
                  k++
                  data_start_idx++
              }
         This writes data point IDs to a contiguous array. Cluster
         boundaries are to be found by compacting length_sums.


    Speed note: this could be improved in step D.vi. by reducing
    thread divergence by only working with clusters that are over
    the threshold...

    Maybe something like:

    1. Use length_sums[] to compact lengths[], producing
       cluster_lengths[]. Simultaneously compact runs[], producing
       cluster_runs[]. The cluster_lengths[i] is the length of the ith
       cluster. cluster_runs[i] is the start index of the ith cluster
       _in the sorted data_.

    2. Exclusive scan the cluster_lengths to produce
       sum_cluster_lengths[]; the kth element of _that_ is the start
       index of the kth cluster in a result dataset.

    3.
      for i in range (0, len(cluster_lengths)):
          n = cluster_lengths[i]
          j = cluster_runs[i]
          offset = sum_cluster_lengths[i]
          k = 0
          for (; j + k < n; k++) {
             out[offset + k] = data[j]
          }

     The result is that out contains runs of cluster
     datapoints. cluster_lengths and sum_cluster_lengths can be used
     to index into out.


   */

fail:
  if (boundaries_d) {
    cudaFree(boundaries_d);
  }
  return cudaSuccess;
}

cudaError_t find_exposure_boundaries(const float *ts_d, int n, int **boundaries_d, int *n_boundaries_d) {
  // Find all the indexes when timestamp changes. This proceeds in three steps:
  // 1. Map ts_d to 1 if ts_d[i] != ts_d[i-1], 0 otherwise.
  // 2. Inclusive prefix-sum the 1s and 0s.
  // 3. Final value in the prefix sum list is the n_boundaries
  // 4. Allocate boundaries_d
  // 5. Extract the boundary points by compacting using the prefix sums.

  cudaError_t err;
  int n_threads = 256;
  int n_blocks = (n + n_threads - 1) / n_threads;
  void *temp_storage_d = NULL;
  size_t temp_storage_bytes = 0;

  // Part 1: Mark changes
  int *changes_d;
  CUDA_OR_FAIL(cudaMalloc((void **)&changes_d, n * sizeof(int)));

  mark_changes_kernel<<<n_blocks, n_threads>>>(ts_d, changes_d, n);
  CUDA_CHECK_ERROR();

  // Part 2: Prefix sum
  int *summed_changes_d;
  CUDA_OR_FAIL(cudaMalloc((void **)&summed_changes_d, n * sizeof(int)));
  // Determine storage requirements for a CUB prefix scan over changes_d
  CUDA_OR_FAIL(cub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes, changes_d, summed_changes_d, n));
  CUDA_OR_FAIL(cudaMalloc(&temp_storage_d, temp_storage_bytes));
  CUDA_OR_FAIL(cub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes, changes_d, summed_changes_d, n));
  cudaFree(temp_storage_d);
  temp_storage_d = NULL;

  // Part 3: Determine n_boundaries
  int n_boundaries_h;

  CUDA_OR_FAIL(cudaMemcpy(&n_boundaries_h, summed_changes_d + n - 1, sizeof(int), cudaMemcpyDeviceToHost));

  assert(n_boundaries_h >= 0);

  CUDA_OR_FAIL(cudaMemcpy(n_boundaries_d, &n_boundaries_h, sizeof(int), cudaMemcpyHostToDevice));

  // Part 4: Allocate boundaries_d
  int *boundaries;
  CUDA_OR_FAIL(cudaMalloc((void **)&boundaries, n_boundaries_h * sizeof(int)));

  // Part 5: Extract boundary points
  gather_change_indexes_kernel<<<n_blocks, n_threads>>>(summed_changes_d, boundaries, n);
  CUDA_CHECK_ERROR();

  *boundaries_d = boundaries;
  err = cudaSuccess;

fail:
  if (changes_d) {
    cudaFree(changes_d);
  }
  if (summed_changes_d) {
    cudaFree(summed_changes_d);
  }
  if (temp_storage_d) {
    cudaFree(temp_storage_d);
  }
  return err;
}

// 1.0 1.0  2.0 3.0 3.0 4.0
// 0   0    1   1   0   1
// 0   0    1   2   2   3
__global__ void mark_changes_kernel(const float *in, int *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0) {
    out[0] = 0;
    i += blockDim.x * gridDim.x;
  }
  for (; i < n; i += blockDim.x * gridDim.x) {
    out[i] = in[i] != in[i - 1];
  }
}

__global__ void gather_change_indexes_kernel(const int *summed_changes, int *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0) {
    out[0] = 0;
    i += blockDim.x * gridDim.x;
  }
  for (; i < n; i += blockDim.x * gridDim.x) {
    if (summed_changes[i] != summed_changes[i - 1]) {
      out[summed_changes[i]] = i;
    }
  }
}