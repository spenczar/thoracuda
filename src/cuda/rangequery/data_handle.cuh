#pragma once

#include <vector>
#include "rangequery/row.cuh"

using thoracuda::rangequery::Row;

namespace thoracuda {
namespace rangequery {

class DataHandle {
  /// Represents a handle to a collection of data suitable for range
  /// queries, stored inside the CUDA device.

public:
  Row *rows;
  int n;

  /// Create a new DataHandle with the given data. It is copied onto the device.
  DataHandle(const std::vector<float> &x, const std::vector<float> &y, const std::vector<float> &t, const std::vector<int> &id);

  /// Frees the memory on the device.
  ~DataHandle();

  /// Copying is not allowed.
  DataHandle(const DataHandle&) = delete;
  DataHandle& operator=(const DataHandle&) = delete;

  /// Moving is allowed.
  
  DataHandle(DataHandle &&other);
  DataHandle& operator=(DataHandle &&other);  
};
  
} // namespace rangequery
} // namespace thoracuda