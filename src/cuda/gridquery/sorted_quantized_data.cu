#include <cuda_runtime.h>

#include <cub/device/device_radix_sort.cuh>
#include <vector>

#include "cuda_macros.h"
#include "gridquery/quantized_data.cuh"
#include "gridquery/sorted_quantized_data.cuh"

using thoracuda::gridquery::QuantizedData;

namespace thoracuda {
namespace gridquery {

SortedQuantizedData::SortedQuantizedData(const QuantizedData &qd) {
  cudaError_t err;
  this->n = qd.n;
  CUDA_OR_THROW(cudaMalloc(&this->sorted_quantized, qd.n * sizeof(int)));
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  CUDA_OR_THROW(
      cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, qd.quantized, this->sorted_quantized, qd.n));
  CUDA_OR_THROW(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  CUDA_OR_THROW(
      cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, qd.quantized, this->sorted_quantized, qd.n));
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

}  // namespace gridquery
}  // namespace thoracuda