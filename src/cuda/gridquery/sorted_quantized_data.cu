#include <cuda_runtime.h>

#include <cub/device/device_radix_sort.cuh>
#include <vector>

#include "cuda_macros.h"
#include "rangequery/data_handle.cuh"
#include "rangequery/row.cuh"
#include "gridquery/quantized_data.cuh"
#include "gridquery/sorted_quantized_data.cuh"

using thoracuda::gridquery::QuantizedData;
using thoracuda::rangequery::Row;
using thoracuda::rangequery::DataHandle;


namespace thoracuda {
namespace gridquery {

  SortedQuantizedData::SortedQuantizedData(const QuantizedData &qd, const DataHandle &dh) {
  cudaError_t err;
  this->n = qd.n;
  CUDA_OR_THROW(cudaMalloc(&this->d_sorted_quantized, qd.n * sizeof(int)));
  CUDA_OR_THROW(cudaMalloc(&this->d_rows, qd.n * sizeof(Row)));
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  CUDA_OR_THROW(
		cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, qd.d_quantized, this->d_sorted_quantized, dh.rows, this->d_rows, qd.n));
  CUDA_OR_THROW(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  CUDA_OR_THROW(
		cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, qd.d_quantized, this->d_sorted_quantized, dh.rows, this->d_rows, qd.n));		
  CUDA_OR_THROW(cudaFree(d_temp_storage));
}

SortedQuantizedData::~SortedQuantizedData() {
  if (this->d_sorted_quantized != nullptr) {
    cudaFree(this->d_sorted_quantized);
  }
  if (this->d_rows != nullptr) {
    cudaFree(this->d_rows);
  }
}

SortedQuantizedData::SortedQuantizedData(SortedQuantizedData &&other) {
  this->d_sorted_quantized = other.d_sorted_quantized;
  this->d_rows = other.d_rows;
  this->n = other.n;
  other.d_sorted_quantized = nullptr;
  other.d_rows = nullptr;
  other.n = 0;
}

SortedQuantizedData &SortedQuantizedData::operator=(SortedQuantizedData &&other) {
  if (this->d_sorted_quantized != nullptr) {
    cudaFree(this->d_sorted_quantized);
  }
  if (this->d_rows != nullptr) {
    cudaFree(this->d_rows);
  }
  
  this->d_sorted_quantized = other.d_sorted_quantized;
  this->d_rows = other.d_rows;
  this->n = other.n;
  other.d_sorted_quantized = nullptr;
  other.d_rows = nullptr;
  other.n = 0;
  return *this;
}

}  // namespace gridquery
}  // namespace thoracuda