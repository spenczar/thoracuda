#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>

#include "cuda_macros.h"
#include "rangequery/offsets_table.cuh"
#include "rangequery/counts_table.cuh"

using thoracuda::rangequery::DataHandle;
using thoracuda::rangequery::CountsTable;

namespace thoracuda {
namespace rangequery {

OffsetsTable::OffsetsTable(const CountsTable &counts) {
  this->initialize_from_counts(counts);
}

void OffsetsTable::initialize_from_counts(const CountsTable &counts) {
  cudaError_t err;
  // Prefix sum to get the offsets
  int *offsets;
  CUDA_OR_THROW(cudaMalloc(&offsets, counts.n_counts * sizeof(int)));
  void *temp_storage_d = NULL;
  size_t temp_storage_bytes = 0;
  CUDA_OR_THROW(cub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes, counts.counts, offsets, counts.n_counts));
  CUDA_OR_THROW(cudaMalloc(&temp_storage_d, temp_storage_bytes));
  CUDA_OR_THROW(cub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes, counts.counts, offsets, counts.n_counts));
  cudaFree(temp_storage_d);
  CUDA_OR_THROW(cudaGetLastError());

  this->offsets = offsets;
  this->n_offsets = counts.n_counts;
}

  OffsetsTable::OffsetsTable(const DataHandle &data, float radius) {
    CountsTable counts(data, radius);
    this->initialize_from_counts(counts);
  }

  OffsetsTable::OffsetsTable() {
    offsets = nullptr;
    n_offsets = 0;
  }
    
  OffsetsTable::~OffsetsTable() {
    if (offsets != nullptr) {
      cudaFree(offsets);
    }
  }

  OffsetsTable::OffsetsTable(OffsetsTable &&other) {
    offsets = other.offsets;
    n_offsets = other.n_offsets;

    other.offsets = nullptr;
    other.n_offsets = 0;
  }

  OffsetsTable& OffsetsTable::operator=(OffsetsTable &&other) {
    if (this != &other) {
      offsets = other.offsets;
      n_offsets = other.n_offsets;

      other.offsets = nullptr;
      other.n_offsets = 0;
    }
    return *this;
  }

    int OffsetsTable::n_total_results() const {
      cudaError_t err;
      int last_offset;
      CUDA_OR_THROW(cudaMemcpy(&last_offset, this->offsets + this->n_offsets - 1, sizeof(int), cudaMemcpyDeviceToHost));
      return last_offset;
    }


}
}