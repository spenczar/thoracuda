#include <cuda_runtime.h>

#include <cub/device/device_scan.cuh>

#include "cuda_macros.h"
#include "gridquery/offsets_grid.cuh"
#include "gridquery/counts_grid.cuh"

using thoracuda::gridquery::CountsGrid;


namespace thoracuda {
namespace gridquery {
  OffsetsGrid::OffsetsGrid(const CountsGrid &counts_grid) {
    cudaError_t err;
    
    this->n_cells = counts_grid.n_cells;

    int n = this->n_cells * this->n_cells;
    CUDA_OR_THROW(cudaMalloc(&this->d_offsets, n * sizeof(int)));

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    CUDA_OR_THROW(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, counts_grid.d_counts, this->d_offsets, n));
    CUDA_OR_THROW(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    CUDA_OR_THROW(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, counts_grid.d_counts, this->d_offsets, n));
    cudaFree(d_temp_storage);
    
    d_temp_storage = nullptr;
    temp_storage_bytes = 0;
  }

  OffsetsGrid::~OffsetsGrid() {
    cudaFree(this->d_offsets);
  }

  OffsetsGrid::OffsetsGrid(OffsetsGrid &&other) {
    this->n_cells = other.n_cells;
    this->d_offsets = other.d_offsets;
    other.d_offsets = nullptr;
  }

  OffsetsGrid &OffsetsGrid::operator=(OffsetsGrid &&other) {
    this->n_cells = other.n_cells;
    this->d_offsets = other.d_offsets;
    other.d_offsets = nullptr;
    return *this;
  }
}
}