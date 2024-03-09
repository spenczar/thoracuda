#pragma once

#include <vector>

#include "pairminmax.h"
#include "rangequery/data_handle.cuh"
#include "rangequery/row.cuh"

using thoracuda::rangequery::DataHandle;

namespace thoracuda {
namespace gridquery {

  struct QuantizedData {
    int *d_quantized;
    XYBounds bounds;
    int n;

    /// Makes an empty quantized data
    QuantizedData();
    
    /// Quantizes data in the data handle
    QuantizedData(DataHandle &dh, int n_cells);
    QuantizedData(DataHandle &dh, int n_cells, struct XYBounds bounds);

    /// Frees the memory allocated for the quantized data
    ~QuantizedData();

    /// Copying is not allowed
    QuantizedData(const QuantizedData &other) = delete;
    QuantizedData &operator=(const QuantizedData &other) = delete;

    /// Moving is allowed
    QuantizedData(QuantizedData &&other);
    QuantizedData &operator=(QuantizedData &&other);

    std::vector<int2> to_host_vector() const;
  };

  __global__ void quantize_data(struct XYPair *pairs, int n, struct XYBounds bounds, int n_cells, int *quantized);
  __global__ void copy_rows_to_xypairs(thoracuda::rangequery::Row *rows, int n, struct XYPair *pairs);
  
}
}