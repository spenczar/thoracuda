// Reimplementation of rangearray.cu to only use C instead of C++
#include <assert.h>

#include <iostream>

#include "cuda_macros.h"
#include "rangearray_c.h"

struct XYBounds xy_bounds_serial(struct XYPair *xys, int n) {
  struct XYBounds bounds;
  bounds.xmin = xys[0].x;
  bounds.xmax = xys[0].x;
  bounds.ymin = xys[0].y;
  bounds.ymax = xys[0].y;
  for (int i = 1; i < n; i++) {
    if (xys[i].x < bounds.xmin) {
      bounds.xmin = xys[i].x;
    }
    if (xys[i].x > bounds.xmax) {
      bounds.xmax = xys[i].x;
    }
    if (xys[i].y < bounds.ymin) {
      bounds.ymin = xys[i].y;
    }
    if (xys[i].y > bounds.ymax) {
      bounds.ymax = xys[i].y;
    }
  }
  return bounds;
}

int xy_bounds_parallel(struct XYPair *xys, int n, struct XYBounds *bounds) {
  // Computes the min and max values of x and y in an array of X-Y
  // pairs.
  //
  // This is done in parallel on the GPU. There are three kernels:
  //
  //  1. Map the xy pairs to {min_x: x, max_x: x, min_y: y, max_y: y}
  //     in a big array of length n.
  //
  //  2. Reduce that big array, block-wise. Each CUDA block computes
  //     the _actual_ min and max values within a chunk of values, and
  //     writes the result into a smaller array of length n_blocks.
  //
  //  3. Reduce the n_block long array into a final answer.
  //
  // The first step is pretty trivial - just a transformation.
  //
  // The second step and third steps are a reduce operation. This
  // happens on a single block, and uses block shared memory as a
  // scratch space. There needs to be one "slot" per thread, where a
  // "slot" is a Bounds struct (size 4 * float).
  cudaError_t err;
  int threads_per_block = 256;
  int nblocks = (n + threads_per_block - 1) / threads_per_block;
  int shared_mem_size = threads_per_block * sizeof(struct XYBounds);
  int npad = threads_per_block - nblocks;

  struct XYBounds *bounds_d = NULL;
  struct XYBounds *bounds_per_block_d = NULL;

  // Copy the xys to the device.
  struct XYPair *xys_d;

  int xys_size = n * sizeof(struct XYPair);
  CUDA_OR_FAIL(cudaMalloc((void **)&xys_d, xys_size));
  CUDA_OR_FAIL(cudaMemcpy(xys_d, xys, xys_size, cudaMemcpyHostToDevice));

  // Parallel reduction to find min/max
  // First, transform the xyvec into a bounds array
  CUDA_OR_FAIL(cudaMalloc((void **)&bounds_d, n * sizeof(struct XYBounds)));

  xyvec_bounds_transform_kernel<<<nblocks, threads_per_block>>>(xys_d, n, bounds_d);
  CUDA_CHECK_ERROR();

  // Next, reduce the bounds array to a single block
  CUDA_OR_FAIL(cudaMalloc((void **)&bounds_per_block_d, threads_per_block * sizeof(struct XYBounds)));
  xyvec_bounds_reduce_kernel<<<nblocks, threads_per_block, shared_mem_size>>>(bounds_d, n, bounds_per_block_d);
  CUDA_CHECK_ERROR();

  // Finally, reduce the last block to a single value. We can reuse
  // bounds_d for this. But first, pad bounds_per_block with
  // INFINITY/-INFINITY
  //
  // If there are more blocks than threads, then this padding is a
  // little more complicated, as is the kernel invocation. That's not
  // implemented yet.
  assert(nblocks <= threads_per_block);

  if (npad > 0) {
    struct XYBounds pad = {INFINITY, -INFINITY, INFINITY, -INFINITY};
    int padsize = npad * sizeof(struct XYBounds);
    struct XYBounds *pad_h = (struct XYBounds *)malloc(padsize);

    for (int i = 0; i < npad; i++) {
      pad_h[i] = pad;
    }
    // Can't use the macro here, because we need to free pad_h
    err = cudaMemcpy((void *)(bounds_per_block_d + nblocks), pad_h, padsize, cudaMemcpyHostToDevice);
    free(pad_h);
    if (err != cudaSuccess) {
      goto fail;
    }
  }

  xyvec_bounds_reduce_kernel<<<1, threads_per_block, shared_mem_size>>>(bounds_per_block_d, nblocks, bounds_d);
  CUDA_CHECK_ERROR();

  CUDA_OR_FAIL(cudaMemcpy(bounds, bounds_d, sizeof(struct XYBounds), cudaMemcpyDeviceToHost));
  CUDA_OR_FAIL(cudaDeviceSynchronize());

fail:
  if (bounds_per_block_d != NULL) {
    cudaFree(bounds_per_block_d);
  }
  if (bounds_d != NULL) {
    cudaFree(bounds_d);
  }
  if (xys_d != NULL) {
    cudaFree(xys_d);
  }
  return err;
}

__global__ void xyvec_bounds_reduce_kernel(struct XYBounds *bounds, int n, struct XYBounds *bounds_per_block) {
  extern __shared__ struct XYBounds tmp[];
  int tid = threadIdx.x;
  tmp[tid] = (tid < n) ? bounds[tid] : (struct XYBounds){INFINITY, -INFINITY, INFINITY, -INFINITY};
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      XYBounds have = tmp[tid];
      XYBounds merge = tmp[tid + stride];
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
    bounds_per_block[blockIdx.x] = tmp[0];
  }
}

__global__ void xyvec_bounds_transform_kernel(struct XYPair *xy, int n, struct XYBounds *bounds) {
  // Map each xy to an XYBounds
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  struct XYBounds b;
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
