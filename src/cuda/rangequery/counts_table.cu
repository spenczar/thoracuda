#include <cuda_runtime.h>

#include "cuda_macros.h"
#include "rangequery/counts_table.cuh"
#include "rangequery/row.cuh"

using thoracuda::rangequery::Row;

namespace thoracuda {
namespace rangequery {
CountsTable::CountsTable() {
  counts = nullptr;
  n_counts = 0;
}

CountsTable::CountsTable(const DataHandle &data, float radius, int threads_per_block) {
  cudaError_t err;

  // Count the number of results for each point
  int *counts;
  CUDA_OR_THROW(cudaMalloc(&counts, data.n * sizeof(int)));

  int blocks = (data.n + threads_per_block - 1) / threads_per_block;

  count_within_distance<<<blocks, threads_per_block>>>(data.rows, data.n, radius * radius, counts);

  this->counts = counts;
  this->n_counts = data.n;
}

CountsTable::~CountsTable() {
  if (counts != nullptr) {
    cudaFree(counts);
  }
}

CountsTable::CountsTable(CountsTable &&other) {
  counts = other.counts;
  n_counts = other.n_counts;

  other.counts = nullptr;
  other.n_counts = 0;
}

CountsTable &CountsTable::operator=(CountsTable &&other) {
  if (this != &other) {
    counts = other.counts;
    n_counts = other.n_counts;

    other.counts = nullptr;
    other.n_counts = 0;
  }
  return *this;
}

__global__ void count_within_distance(Row *rows, int n, float sq_distance, int *result) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for (; i < n; i += blockDim.x * gridDim.x) {
    int count = 0;
    Row row_i = rows[i];
    for (int j = 0; j < n; j++) {
      Row row_j = rows[j];
      if (row_i.t == row_j.t) {
        continue;
      }
      float dx = row_i.x - row_j.x;
      float dy = row_i.y - row_j.y;
      float sq_dist = dx * dx + dy * dy;
      if (sq_dist <= sq_distance) {
        count++;
      }
    }
    result[i] = count;
  }
}

}  // namespace rangequery
}  // namespace thoracuda