#pragma once

#include <cuda_runtime.h>

#include "pair.h"

// Error definitions
#define ERR_CUDA_MALLOC 1
#define ERR_CUDA_MEMCPY 2
#define ERR_MALLOC 3

// Structures
struct XYBounds {
  float xmin;
  float xmax;
  float ymin;
  float ymax;
};

// Function prototypes
struct XYBounds xy_bounds_serial(struct XYPair *xys, int n);
cudaError_t xy_bounds_parallel(struct XYPair *xys, int n, struct XYBounds *bounds);
cudaError_t xy_bounds_parallel_on_device(struct XYPair *xys, int n, struct XYBounds *bounds);
__global__ void xyvec_bounds_transform_kernel(struct XYPair *xy, int n, struct XYBounds *bounds);
__global__ void xyvec_bounds_reduce_kernel(struct XYBounds *bounds, int n, struct XYBounds *bounds_per_block);
