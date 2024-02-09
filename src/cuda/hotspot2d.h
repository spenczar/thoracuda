#pragma once
#include <cuda_runtime.h>

#include "pair.h"

struct Cluster {
  int *point_ids;
  int n;
  float v_x;
  float v_y;
};

struct Hotspot2DParams {
  // Minimum number of points in a cluster.
  int min_points;

  // Maximum distance between points in a cluster.
  float max_distance;

  // Maximum (absolute value) relative velocity when probing for
  // clusters, measured in gnomonic degrees.
  float max_rel_velocity;
};

/*
  The hotspot2d algorithm finds 'clusters' of related points in a 2D
  grid using a naive fixed-width grid.
 */
cudaError_t hotspot2d_parallel(const struct XYPair *xys, const float *ts, const int *ids, int n,
                               struct Hotspot2DParams params, struct Cluster **clusters, int *n_clusters);

// Like hotspot2d_parallel, but running on the GPU. Not responsible
// for copying memory onto or off of the device.
cudaError_t _hotspot2d_parallel_on_device(const struct XYPair *xys_d, const float *ts_d, const int *ids_d, int n,
                                          struct Hotspot2DParams params, struct Cluster **clusters_d,
                                          int *n_clusters_d);

cudaError_t find_exposure_boundaries(const float *ts_d, int n, int **boundaries_d, int *n_boundaries_d);
__global__ void mark_changes_kernel(const float *in, int *out, int n);
__global__ void gather_change_indexes_kernel(const int *summed_changes, int *out, int n);
