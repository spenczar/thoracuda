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

  SortedQuantizedData::SortedQuantizedData(const QuantizedData &qd) {
    cudaError_t err;
    this->n = qd.n;
    CUDA_OR_THROW(cudaMalloc(&this->sorted_quantized, qd.n * sizeof(int)));
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    CUDA_OR_THROW(cub::DeviceRadixSort::SortKeys(d_temp_storage,
						 temp_storage_bytes,
						 qd.quantized,
						 this->sorted_quantized,
						 qd.n));
    CUDA_OR_THROW(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    CUDA_OR_THROW(cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, qd.quantized, this->sorted_quantized, qd.n));
    CUDA_OR_THROW(cudaFree(d_temp_storage));
  }

  SortedQuantizedData::~SortedQuantizedData() {
    if (this->sorted_quantized != nullptr) {
      cudaFree(this->sorted_quantized);
    }
  }

  SortedQuantizedData::SortedQuantizedData(SortedQuantizedData &&other) {
    this->sorted_quantized = other.sorted_quantized;
    this->n = other.n;
    other.sorted_quantized = nullptr;
    other.n = 0;
  }

  SortedQuantizedData &SortedQuantizedData::operator=(SortedQuantizedData &&other) {
    if (this->sorted_quantized != nullptr) {
      cudaFree(this->sorted_quantized);
    }
    this->sorted_quantized = other.sorted_quantized;
    this->n = other.n;
    other.sorted_quantized = nullptr;
    other.n = 0;
    return *this;
  }

  /* COUNTS GRID */

  CountsGrid::CountsGrid(int n_cells, struct XYBounds bounds) {
    cudaError_t err;
    this->n_cells = n_cells;
    this->bounds = bounds;
    
    CUDA_OR_THROW(cudaMalloc(&this->counts, n_cells * n_cells * sizeof(int)));
    CUDA_OR_THROW(cudaMemset(this->counts, 0, n_cells * n_cells * sizeof(int)));
  }


  CountsGrid::CountsGrid(const DataHandle &dh, const SortedQuantizedData &sqd, int n_cells, struct XYBounds bounds) {
    cudaError_t err;
    this->n_cells = n_cells;
    this->bounds = bounds;

    CUDA_OR_THROW(cudaMalloc(&this->counts, n_cells * n_cells * sizeof(int)));
    CUDA_OR_THROW(cudaMemset(this->counts, 0, n_cells * n_cells * sizeof(int)));
    
    // 1. Count the number of occurrences of each cell.

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    // The CUB run length encoder provides us with the unique keys and
    // the run lengths. We don't actually use the unique keys.
    int *run_keys_discarded = nullptr;
    int *run_lengths = nullptr;
    int *num_runs = nullptr;
    CUDA_OR_THROW(cudaMalloc(&run_keys_discarded, sqd.n * sizeof(int)));
    CUDA_OR_THROW(cudaMalloc(&run_lengths, sqd.n * sizeof(int)));
    CUDA_OR_THROW(cudaMalloc(&num_runs, sizeof(int)));
    
    CUDA_OR_THROW(cub::DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes, sqd.sorted_quantized, run_keys_discarded, run_lengths, num_runs, sqd.n));
    CUDA_OR_THROW(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    CUDA_OR_THROW(cub::DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes, sqd.sorted_quantized, run_keys_discarded, run_lengths, num_runs, sqd.n));
    
    CUDA_OR_THROW(cudaFree(d_temp_storage));
    CUDA_OR_THROW(cudaFree(run_keys_discarded));
    
    d_temp_storage = nullptr;
    temp_storage_bytes = 0;

    int num_runs_host = 0;
    CUDA_OR_THROW(cudaMemcpy(&num_runs_host, num_runs, sizeof(int), cudaMemcpyDeviceToHost));

    // 2. Prefix sum the run lengths to get run offsets
    int *run_offsets = nullptr;
    CUDA_OR_THROW(cudaMalloc(&run_offsets, (num_runs_host + 1) * sizeof(int)));
    CUDA_OR_THROW(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, run_lengths, run_offsets, num_runs_host));
    CUDA_OR_THROW(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    CUDA_OR_THROW(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, run_lengths, run_offsets, num_runs_host));
    
    // 3. Map the run lengths to the counts grid.
    int n_threads = 256;
    int n_blocks = (num_runs_host + n_threads - 1) / n_threads;
    map_cell_counts<<<n_blocks, n_threads>>>(this->counts, n_cells, sqd.sorted_quantized, run_offsets, run_lengths, num_runs_host);
    CUDA_OR_THROW(cudaGetLastError());
    
    cudaFree(run_offsets);
    cudaFree(run_lengths);
  }

  std::vector<std::vector<int>> CountsGrid::to_host_vector() const {
    cudaError_t err;
    
    int n = this->n_cells * this->n_cells;
    int *host_counts = new int[n];

    CUDA_OR_THROW(cudaMemcpy(host_counts, this->counts, n * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<std::vector<int>> result;
    result.reserve(this->n_cells);
    for (int i = 0; i < this->n_cells; i++) {
      std::vector<int> row;
      row.reserve(this->n_cells);
      for (int j = 0; j < this->n_cells; j++) {
	row.push_back(host_counts[i * this->n_cells + j]);
      }
      result.push_back(row);
    }

    delete[] host_counts;
    return result;
  }

  CountsGrid::~CountsGrid() {
    if (this->counts != nullptr) {
      cudaFree(this->counts);
    }
  }

  CountsGrid::CountsGrid(CountsGrid &&other) {
    this->counts = other.counts;
    this->n_cells = other.n_cells;
    this->bounds = other.bounds;
    other.counts = nullptr;
  }

  CountsGrid &CountsGrid::operator=(CountsGrid &&other) {
    if (this->counts != nullptr) {
      cudaFree(this->counts);
    }
    this->counts = other.counts;
    this->n_cells = other.n_cells;
    this->bounds = other.bounds;
    other.counts = nullptr;
    return *this;
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