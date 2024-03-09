#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cuda/std/tuple>
#include <stdexcept>

#include "cuda_macros.h"
#include "gridquery/quantized_data.cuh"
#include "pair.h"
#include "pairminmax.h"
#include "rangequery/data_handle.cuh"
#include "rangequery/row.cuh"

using thoracuda::rangequery::DataHandle;

namespace thoracuda {
namespace gridquery {

QuantizedData::QuantizedData() {
  this->d_quantized = nullptr;
  this->n = 0;
}

QuantizedData::QuantizedData(DataHandle &dh, int n_cells) {
  cudaError_t err;
  int n_threads = 256;
  int n_blocks = (dh.n + n_threads - 1) / n_threads;

  // First, establish bounds of the data.
  struct XYPair *pairs = nullptr;

  CUDA_OR_FAIL(cudaMalloc(&pairs, dh.n * sizeof(struct XYPair)));
  copy_rows_to_xypairs<<<n_blocks, n_threads>>>(dh.rows, dh.n, pairs);
  CUDA_OR_FAIL(cudaGetLastError());

  // Now, find the min and max of the data.
  CUDA_OR_FAIL(xy_bounds_parallel(pairs, dh.n, &(this->bounds)));

  // Now, quantize the data.
  CUDA_OR_FAIL(cudaMalloc(&this->d_quantized, dh.n * sizeof(int2)));
  quantize_data<<<n_blocks, n_threads>>>(pairs, dh.n, this->bounds, n_cells, this->d_quantized);
  cudaDeviceSynchronize();
  CUDA_OR_FAIL(cudaGetLastError());
  this->n = dh.n;

fail:
  if (pairs != nullptr) {
    cudaFree(pairs);
  }
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }
}

QuantizedData::QuantizedData(DataHandle &dh, int n_cells, struct XYBounds bounds) {
  cudaError_t err;
  int n_threads = 256;
  int n_blocks = (dh.n + n_threads - 1) / n_threads;

  CUDA_OR_THROW(cudaMalloc(&this->d_quantized, dh.n * sizeof(int2)));

  // Copy the rows to a new array of XYPairs.
  struct XYPair *pairs = nullptr;
  CUDA_OR_FAIL(cudaMalloc(&pairs, dh.n * sizeof(struct XYPair)));
  copy_rows_to_xypairs<<<n_blocks, n_threads>>>(dh.rows, dh.n, pairs);
  CUDA_OR_THROW(cudaGetLastError());

  // Quantize the data.
  quantize_data<<<n_blocks, n_threads>>>(pairs, dh.n, bounds, n_cells, this->d_quantized);

  cudaDeviceSynchronize();
  CUDA_OR_THROW(cudaGetLastError());

  this->n = dh.n;

fail:
  if (pairs != nullptr) {
    cudaFree(pairs);
  }
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }
}

QuantizedData::~QuantizedData() {
  if (this->d_quantized != nullptr) {
    cudaFree(this->d_quantized);
  }
}

QuantizedData::QuantizedData(QuantizedData &&other) {
  this->d_quantized = other.d_quantized;
  this->n = other.n;
  other.d_quantized = nullptr;
  other.n = 0;
}

QuantizedData &QuantizedData::operator=(QuantizedData &&other) {
  if (this->d_quantized != nullptr) {
    cudaFree(this->d_quantized);
  }
  this->d_quantized = other.d_quantized;
  this->n = other.n;
  other.d_quantized = nullptr;
  other.n = 0;
  return *this;
}

std::vector<int2> QuantizedData::to_host_vector() const {
  cudaError_t err;
  std::vector<int2> result;
  result.reserve(this->n);
  int *host_quantized = new int[this->n];

  CUDA_OR_THROW(cudaMemcpy(host_quantized, this->d_quantized, this->n * sizeof(int), cudaMemcpyDeviceToHost));
  for (int i = 0; i < this->n; i++) {
    int x = host_quantized[i] >> 16;
    int y = host_quantized[i] & 0xFFFF;
    result.push_back(make_int2(x, y));
  }
  delete[] host_quantized;
  return result;
}

__global__ void copy_rows_to_xypairs(thoracuda::rangequery::Row *rows, int n, struct XYPair *pairs) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for (; i < n; i += blockDim.x * gridDim.x) {
    pairs[i].x = rows[i].x;
    pairs[i].y = rows[i].y;
  }
}

__global__ void quantize_data(struct XYPair *pairs, int n, struct XYBounds bounds, int n_cells, int *quantized) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  for (; i < n; i += blockDim.x * gridDim.x) {
    float x = pairs[i].x;
    float y = pairs[i].y;
    int cell_x = (x - bounds.xmin) / (bounds.xmax - bounds.xmin) * n_cells;
    int cell_y = (y - bounds.ymin) / (bounds.ymax - bounds.ymin) * n_cells;
    quantized[i] = cell_x << 16 | cell_y;
  }
}
}  // namespace gridquery
}  // namespace thoracuda