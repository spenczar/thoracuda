#include <stdexcept>

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>
#include <cuda/std/tuple>

#include "gridquery/quantized_data.cuh"

#include "pair.h"
#include "pairminmax.h"
#include "rangequery/data_handle.cuh"
#include "rangequery/row.cuh"
#include "cuda_macros.h"

using thoracuda::rangequery::DataHandle;

namespace thoracuda {
namespace gridquery {

  QuantizedData::QuantizedData() {
    this->quantized = nullptr;
    this->n = 0;
  }

  QuantizedData::QuantizedData(DataHandle &dh, int n_cells) {
    cudaError_t err;
    int n_threads = 256;
    int n_blocks = (dh.n + n_threads - 1) / n_threads;
    
    // First, establish bounds of the data.
    struct XYBounds *bounds = nullptr;
    struct XYPair *pairs = nullptr;
    struct XYBounds *host_bounds = new struct XYBounds;
    
    CUDA_OR_FAIL(cudaMalloc(&pairs, dh.n * sizeof(struct XYPair)));
    copy_rows_to_xypairs<<<n_blocks, n_threads>>>(dh.rows, dh.n, pairs);
    CUDA_OR_FAIL(cudaGetLastError());

    // Now, find the min and max of the data.
    CUDA_OR_FAIL(xy_bounds_parallel(pairs, dh.n, host_bounds));

    // Now, quantize the data.
    CUDA_OR_FAIL(cudaMalloc(&this->quantized, dh.n * sizeof(int2)));
    quantize_data<<<n_blocks, n_threads>>>(pairs, dh.n, *host_bounds, n_cells, this->quantized);
    cudaDeviceSynchronize();
    CUDA_OR_FAIL(cudaGetLastError());
    this->n = dh.n;

  fail:
    if (bounds != nullptr) {
      delete bounds;
    }
    if (pairs != nullptr) {
      cudaFree(pairs);
    }
    if (host_bounds != nullptr) {
      delete host_bounds;
    }
    if (err != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(err));
    }
  }

  QuantizedData::QuantizedData(DataHandle &dh, int n_cells, struct XYBounds bounds) {
    cudaError_t err;
    int n_threads = 256;
    int n_blocks = (dh.n + n_threads - 1) / n_threads;
    
    CUDA_OR_THROW(cudaMalloc(&this->quantized, dh.n * sizeof(int2)));

    // Copy the rows to a new array of XYPairs.
    struct XYPair *pairs = nullptr;
    CUDA_OR_FAIL(cudaMalloc(&pairs, dh.n * sizeof(struct XYPair)));
    copy_rows_to_xypairs<<<n_blocks, n_threads>>>(dh.rows, dh.n, pairs);
    CUDA_OR_THROW(cudaGetLastError());
    
    // Quantize the data.
    quantize_data<<<n_blocks, n_threads>>>(pairs, dh.n, bounds, n_cells, this->quantized);
    
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
    if (this->quantized != nullptr) {
      cudaFree(this->quantized);
    }
  }

  QuantizedData::QuantizedData(QuantizedData &&other) {
    this->quantized = other.quantized;
    this->n = other.n;
    other.quantized = nullptr;
    other.n = 0;
  }

  QuantizedData &QuantizedData::operator=(QuantizedData &&other) {
    if (this->quantized != nullptr) {
      cudaFree(this->quantized);
    }
    this->quantized = other.quantized;
    this->n = other.n;
    other.quantized = nullptr;
    other.n = 0;
    return *this;
  }

  std::vector<int2> QuantizedData::to_host_vector() const {
    cudaError_t err;
    std::vector<int2> result;
    result.reserve(this->n);
    int *host_quantized = new int[this->n];
    
    CUDA_OR_THROW(cudaMemcpy(host_quantized, this->quantized, this->n * sizeof(int), cudaMemcpyDeviceToHost));
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
}
}