#include <stdexcept>

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>
#include <cuda/std/tuple>

#include "gridquery/gridquery.cuh"

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


  /* COUNTS GRID */

  CountsGrid::CountsGrid(int n_cells, struct XYBounds bounds) {
    cudaError_t err;
    this->n_cells = n_cells;
    this->bounds = bounds;
    
    CUDA_OR_THROW(cudaMalloc(&this->counts, n_cells * n_cells * sizeof(int)));
    CUDA_OR_THROW(cudaMemset(this->counts, 0, n_cells * n_cells * sizeof(int)));
  }


  CountsGrid::CountsGrid(const DataHandle &dh, const QuantizedData &qd, int n_cells, struct XYBounds bounds) {
    cudaError_t err;
    this->n_cells = n_cells;
    this->bounds = bounds;

    CUDA_OR_THROW(cudaMalloc(&this->counts, n_cells * n_cells * sizeof(int)));
    CUDA_OR_THROW(cudaMemset(this->counts, 0, n_cells * n_cells * sizeof(int)));

    // 1. Sort the quantized data (using CUB).
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    int *sorted_quantized = nullptr;
    CUDA_OR_THROW(cudaMalloc(&sorted_quantized, qd.n * sizeof(int)));


    CUDA_OR_THROW(cub::DeviceRadixSort::SortKeys(d_temp_storage,
						 temp_storage_bytes,
						 qd.quantized,
						 sorted_quantized,
						 dh.n));
    CUDA_OR_THROW(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    CUDA_OR_THROW(cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, qd.quantized, sorted_quantized, dh.n));
    CUDA_OR_THROW(cudaFree(d_temp_storage));
    d_temp_storage = nullptr;
    temp_storage_bytes = 0;
    
    // 2. Count the number of occurrences of each cell.
    int *run_offsets = nullptr;
    int *run_lengths = nullptr;
    int num_runs = 0;
    CUDA_OR_THROW(cub::DeviceRunLengthEncode::NonTrivialRuns(d_temp_storage, temp_storage_bytes, sorted_quantized, run_offsets, run_lengths, &num_runs, qd.n));
    CUDA_OR_THROW(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    CUDA_OR_THROW(cub::DeviceRunLengthEncode::NonTrivialRuns(d_temp_storage, temp_storage_bytes, sorted_quantized, run_offsets, run_lengths, &num_runs, qd.n));
    CUDA_OR_THROW(cudaFree(d_temp_storage));
    d_temp_storage = nullptr;
    temp_storage_bytes = 0;
    
    // 3. Map the run lengths to the counts grid.
    int n_threads = 256;
    int n_blocks = (num_runs + n_threads - 1) / n_threads;
    map_cell_counts<<<n_blocks, n_threads>>>(this->counts, n_cells, sorted_quantized, run_offsets, run_lengths, num_runs);

    cudaFree(run_offsets);
    cudaFree(run_lengths);
    cudaFree(sorted_quantized);
  }

  __global__ void map_cell_counts(int *counts, int n_cells, int *quantized, int *run_offsets, int *run_lengths, int num_runs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_runs) {
      return;
    }
    for (; i < num_runs; i += blockDim.x * gridDim.x) {
      int cell = quantized[run_offsets[i]];
      int x = cell >> 16;
      int y = cell & 0xFFFF;
      atomicAdd(&counts[x * n_cells + y], run_lengths[i]);
    }
  }
}
}