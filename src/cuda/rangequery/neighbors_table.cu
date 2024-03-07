#include <cuda_runtime.h>

#include "cuda_macros.h"
#include "rangequery/data_handle.cuh"
#include "rangequery/neighbors_table.cuh"
#include "rangequery/offsets_table.cuh"
#include "rangequery/row.cuh"

using thoracuda::rangequery::DataHandle;
using thoracuda::rangequery::OffsetsTable;
using thoracuda::rangequery::Row;

namespace thoracuda {
namespace rangequery {

NeighborsTable::NeighborsTable() {
  neighbors = nullptr;
  n_neighbors = 0;
}

NeighborsTable::NeighborsTable(const DataHandle &data, float radius, const OffsetsTable &offsets) {
  cudaError_t err;

  // Count the number of results for each point
  int *neighbors;
  this->n_neighbors = offsets.n_total_results();
  CUDA_OR_THROW(cudaMalloc(&neighbors, this->n_neighbors * sizeof(int)));

  int threads_per_block = 512;
  int blocks = (data.n + threads_per_block - 1) / threads_per_block;

  range_query<<<blocks, threads_per_block>>>(data.rows, data.n, (radius * radius), offsets.offsets, neighbors);
  CUDA_OR_THROW(cudaGetLastError());

  this->neighbors = neighbors;
}

NeighborsTable::~NeighborsTable() {
  if (neighbors != nullptr) {
    cudaFree(neighbors);
  }
}

NeighborsTable::NeighborsTable(NeighborsTable &&other) {
  neighbors = other.neighbors;
  n_neighbors = other.n_neighbors;

  other.neighbors = nullptr;
  other.n_neighbors = 0;
}

NeighborsTable &NeighborsTable::operator=(NeighborsTable &&other) {
  if (this != &other) {
    neighbors = other.neighbors;
    n_neighbors = other.n_neighbors;

    other.neighbors = nullptr;
    other.n_neighbors = 0;
  }
  return *this;
}

std::vector<std::vector<int>> NeighborsTable::get_neighbors(const OffsetsTable &offsets) const {
  cudaError_t err;

  std::vector<std::vector<int>> results;
  results.resize(offsets.n_offsets);

  int *host_results = new int[n_neighbors];
  CUDA_OR_THROW(cudaMemcpy(host_results, this->neighbors, n_neighbors * sizeof(int), cudaMemcpyDeviceToHost));

  int *host_offsets = new int[offsets.n_offsets];
  CUDA_OR_THROW(cudaMemcpy(host_offsets, offsets.offsets, offsets.n_offsets * sizeof(int), cudaMemcpyDeviceToHost));

  for (int i = 0; i < offsets.n_offsets; i++) {
    // Subtle: this indexing scheme is tightly linked to use of
    // exclusive vs inclusive scan
    int start = i == 0 ? 0 : host_offsets[i - 1];
    int end = host_offsets[i];
    results[i].resize(end - start);
    for (int j = start; j < end; j++) {
      results[i][j - start] = host_results[j];
    }
  }
  delete[] host_results;
  return results;
}

__global__ void range_query(Row *rows, int n, float sq_distance, int *offsets, int *result) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for (; i < n; i += blockDim.x * gridDim.x) {
    int count = 0;
    Row row_i = rows[i];
    // Subtle: this indexing scheme is tightly linked to use of
    // exclusive vs inclusive scan
    int start = i == 0 ? 0 : offsets[i - 1];
    for (int j = 0; j < n; j++) {
      Row row_j = rows[j];
      if (row_j.t == row_i.t) {
        continue;
      }
      float dx = row_i.x - row_j.x;
      float dy = row_i.y - row_j.y;
      float sq_dist = dx * dx + dy * dy;
      if (sq_dist <= sq_distance) {
        result[start + count] = row_j.id;
        count++;
      }
    }
  }
}
}  // namespace rangequery
}  // namespace thoracuda