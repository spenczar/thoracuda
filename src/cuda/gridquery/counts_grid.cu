#include <cuda_runtime.h>

#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <vector>

#include "cuda_macros.h"
#include "gridquery/counts_grid.cuh"
#include "gridquery/sorted_quantized_data.cuh"
#include "pairminmax.h"
#include "rangequery/data_handle.cuh"

using thoracuda::gridquery::SortedQuantizedData;
using thoracuda::rangequery::DataHandle;

namespace thoracuda {
namespace gridquery {

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

  CUDA_OR_THROW(cub::DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes, sqd.sorted_quantized,
                                                   run_keys_discarded, run_lengths, num_runs, sqd.n));
  CUDA_OR_THROW(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  CUDA_OR_THROW(cub::DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes, sqd.sorted_quantized,
                                                   run_keys_discarded, run_lengths, num_runs, sqd.n));

  CUDA_OR_THROW(cudaFree(d_temp_storage));
  CUDA_OR_THROW(cudaFree(run_keys_discarded));

  d_temp_storage = nullptr;
  temp_storage_bytes = 0;

  int num_runs_host = 0;
  CUDA_OR_THROW(cudaMemcpy(&num_runs_host, num_runs, sizeof(int), cudaMemcpyDeviceToHost));

  // 2. Prefix sum the run lengths to get run offsets
  int *run_offsets = nullptr;
  CUDA_OR_THROW(cudaMalloc(&run_offsets, (num_runs_host + 1) * sizeof(int)));
  CUDA_OR_THROW(
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, run_lengths, run_offsets, num_runs_host));
  CUDA_OR_THROW(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  CUDA_OR_THROW(
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, run_lengths, run_offsets, num_runs_host));

  // 3. Map the run lengths to the counts grid.
  int n_threads = 256;
  int n_blocks = (num_runs_host + n_threads - 1) / n_threads;
  map_cell_counts<<<n_blocks, n_threads>>>(this->counts, n_cells, sqd.sorted_quantized, run_offsets, run_lengths,
                                           num_runs_host);
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

__global__ void map_cell_counts(int *counts, int n_cells, int *quantized, int *run_offsets, int *run_lengths,
                                int num_runs) {
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
}  // namespace gridquery
}  // namespace thoracuda