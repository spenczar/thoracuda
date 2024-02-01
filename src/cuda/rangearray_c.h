#ifndef RANGEARRAY_CU_H
#define RANGEARRAY_CU_H

#include <cuda_runtime.h>

// Error definitions
#define ERR_CUDA_MALLOC 1
#define ERR_CUDA_MEMCPY 2
#define ERR_MALLOC 3

// Structures
struct Exposure {
  float *x;
  float *y;
  int n;
  double t;  // MJD
};

struct XYPair {
  float x;
  float y;
};

struct XYPairVector {
  struct XYPair *xy;
  int n;
};

int xyvec_init(struct XYPairVector *xyvec, int n);
void xyvec_free(struct XYPairVector *xyvec);
void xyvec_cuda_free(struct XYPairVector *xyvec);

struct XYVectorBounds {
  float xmin;
  float xmax;
  float ymin;
  float ymax;
};

// Function prototypes
int copy_xyvec_to_device(struct XYPairVector *xyvec_h, struct XYPairVector *xyvec_d);
struct XYVectorBounds xyvec_bounds(struct XYPairVector *xyvec_h);
int xyvec_bounds_parallel(struct XYPairVector *xyvec_h, struct XYVectorBounds *bounds);
__global__ void xyvec_bounds_reduce_kernel(struct XYVectorBounds *bounds, int n, struct XYVectorBounds *bounds_per_block);
__global__ void xyvec_bounds_transform_kernel(struct XYPair *xy, int n, struct XYVectorBounds *bounds);
int build_grid_coord_map(struct XYPairVector *xyvec_h, int n);

#endif  // RANGEARRAY_CU_H
