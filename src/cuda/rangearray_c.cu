// Reimplementation of rangearray.cu to only use C instead of C++
#include <iostream>

#include "rangearray_c.h"

int copy_xyvec_to_device(struct XYPairVector *xyvec_h, struct XYPairVector *xyvec_d) {
  int err;

  struct XYPair *xy_d = NULL;
  err = cudaMalloc((void **)&xy_d, (size_t)(xyvec_h->n * sizeof(struct XYPair)));
  if (err != cudaSuccess) {
    return ERR_CUDA_MALLOC;
  }

  err = cudaMemcpy((void *)xy_d, xyvec_h->xy, xyvec_h->n * sizeof(struct XYPair), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(xy_d);
    return ERR_CUDA_MEMCPY;
  }
  xyvec_d->xy = xy_d;
  xyvec_d->n = xyvec_h->n;
  return 0;
}

struct XYPairVector xyvec_create() {
  struct XYPairVector xyvec;
  xyvec.n = 0;
  xyvec.xy = NULL;
  return xyvec;
}

int xyvec_init(struct XYPairVector *xyvec, int n) {
  xyvec->n = n;
  xyvec->xy = (struct XYPair *)malloc(n * sizeof(struct XYPair));
  if (xyvec->xy == NULL) {
    return ERR_MALLOC;
  }
  return 0;
}

void xyvec_free(struct XYPairVector *xyvec) {
  free(xyvec->xy);
  xyvec->xy = NULL;
  xyvec->n = 0;
}

void xyvec_cuda_free(struct XYPairVector *xyvec) {
  if (xyvec->xy != NULL) {
    cudaFree(xyvec->xy);
  }
  xyvec->xy = NULL;
}

struct XYVectorBounds xyvec_bounds(struct XYPairVector *xyvec_h) {
  struct XYVectorBounds bounds;
  bounds.xmin = xyvec_h->xy[0].x;
  bounds.xmax = xyvec_h->xy[0].x;
  bounds.ymin = xyvec_h->xy[0].y;
  bounds.ymax = xyvec_h->xy[0].y;
  for (int i = 1; i < xyvec_h->n; i++) {
    if (xyvec_h->xy[i].x < bounds.xmin) {
      bounds.xmin = xyvec_h->xy[i].x;
    }
    if (xyvec_h->xy[i].x > bounds.xmax) {
      bounds.xmax = xyvec_h->xy[i].x;
    }
    if (xyvec_h->xy[i].y < bounds.ymin) {
      bounds.ymin = xyvec_h->xy[i].y;
    }
    if (xyvec_h->xy[i].y > bounds.ymax) {
      bounds.ymax = xyvec_h->xy[i].y;
    }
  }
  return bounds;
}

int xyvec_bounds_parallel(struct XYPairVector *xyvec_h, struct XYVectorBounds *bounds) {
  int err = 0;
  int threads_per_block = 256;
  int nblocks = (xyvec_h->n + threads_per_block - 1) / threads_per_block;
  int shared_mem_size = threads_per_block * sizeof(struct XYVectorBounds);
  int npad = threads_per_block - nblocks;

  struct XYVectorBounds *bounds_d = NULL;
  struct XYVectorBounds *bounds_per_block_d = NULL;
  struct XYPairVector xyvec_d;

  err = copy_xyvec_to_device(xyvec_h, &xyvec_d);
  if (err != 0) {
    goto fail;
  }

  // Parallel reduction to find min/max
  // First, transform the xyvec into a bounds array
  err = cudaMalloc((void **)&bounds_d, xyvec_d.n * sizeof(struct XYVectorBounds));
  if (err != cudaSuccess) {
    err = ERR_CUDA_MALLOC;
    goto fail;
  }
  xyvec_bounds_transform_kernel<<<nblocks, threads_per_block>>>(xyvec_d.xy, xyvec_d.n, bounds_d);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    goto fail;
  }
  
  // Next, reduce the bounds array to a single block
  err = cudaMalloc((void **)&bounds_per_block_d, threads_per_block * sizeof(struct XYVectorBounds));
  if (err != cudaSuccess) {
    err = ERR_CUDA_MALLOC;
    goto fail;
  }

  xyvec_bounds_reduce_kernel<<<nblocks, threads_per_block, shared_mem_size>>>(bounds_d, xyvec_d.n, bounds_per_block_d);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    goto fail;
  }

  // Finally, reduce the last block to a single value. We can reuse bounds_d for this.
  // But first, pad bounds_per_block with INFINITY/-INFINITY
  if (npad > 0) {
    struct XYVectorBounds pad = {INFINITY, -INFINITY, INFINITY, -INFINITY};
    struct XYVectorBounds *pad_h = (struct XYVectorBounds *)malloc(npad * sizeof(struct XYVectorBounds));
    
    for (int i = 0; i < npad; i++) {
      pad_h[i] = pad;
    }
    err = cudaMemcpy((void *)(bounds_per_block_d + nblocks), pad_h, npad * sizeof(struct XYVectorBounds), cudaMemcpyHostToDevice);
    free(pad_h);
    if (err != cudaSuccess) {
      err = ERR_CUDA_MEMCPY;
      goto fail;
    }
  }

  xyvec_bounds_reduce_kernel<<<1, threads_per_block, shared_mem_size>>>(bounds_per_block_d, nblocks, bounds_d);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    goto fail;
  }

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    goto fail;
  }

  err = cudaMemcpy(bounds, bounds_d, sizeof(struct XYVectorBounds), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    err = ERR_CUDA_MEMCPY;
    goto fail;
  }
  
  err = 0;
fail:
  if (bounds_per_block_d != NULL) {
    cudaFree(bounds_per_block_d);
  }
  if (bounds_d != NULL) {
    cudaFree(bounds_d);
  }
  xyvec_cuda_free(&xyvec_d);
  return err;
}

__global__ void xyvec_bounds_reduce_kernel(struct XYVectorBounds *bounds, int n, struct XYVectorBounds *bounds_per_block) {
  extern __shared__ struct XYVectorBounds tmp[];
  int tid = threadIdx.x;
  tmp[tid] = (tid < n) ? bounds[tid] : (struct XYVectorBounds){INFINITY, -INFINITY, INFINITY, -INFINITY};
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      XYVectorBounds have = tmp[tid];
      XYVectorBounds merge = tmp[tid + stride];
      if (merge.xmin < have.xmin) {
	have.xmin = merge.xmin;
      }
      if (merge.xmax > have.xmax) {
	have.xmax = merge.xmax;
      }
      if (merge.ymin < have.ymin) {
	have.ymin = merge.ymin;
      }
      if (merge.ymax > have.ymax) {
	have.ymax = merge.ymax;
      }
      tmp[tid] = have;
    }
    __syncthreads();
  }

  if (tid == 0) {
    // debug printing
    bounds_per_block[blockIdx.x] = tmp[0];
  }
}

__global__ void xyvec_bounds_transform_kernel(struct XYPair *xy, int n, struct XYVectorBounds *bounds) {
  // Map each xy to an XYVectorBounds
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  struct XYVectorBounds b;
  if (i >= n) {
    b.xmin = INFINITY;
    b.xmax = -INFINITY;
    b.ymin = INFINITY;
    b.ymax = -INFINITY;
  } else {
    b.xmin = xy[i].x;
    b.xmax = xy[i].x;
    b.ymin = xy[i].y;
    b.ymax = xy[i].y;
  }
  bounds[i] = b;
}
