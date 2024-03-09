#pragma once

#include <vector>

#include "rangequery/data_handle.cuh"
#include "rangequery/row.cuh"

#include "gridquery/quantized_data.cuh"

using thoracuda::gridquery::QuantizedData;
using thoracuda::rangequery::Row;
using thoracuda::rangequery::DataHandle;

namespace thoracuda {
namespace gridquery {

  struct SortedQuantizedData {
    int *d_sorted_quantized;
    Row *d_rows;
    int n;

    SortedQuantizedData(const QuantizedData &qd, const DataHandle &dh);

    ~SortedQuantizedData();

    SortedQuantizedData(const SortedQuantizedData &other) = delete;
    SortedQuantizedData &operator=(const SortedQuantizedData &other) = delete;

    SortedQuantizedData(SortedQuantizedData &&other);
    SortedQuantizedData &operator=(SortedQuantizedData &&other);

    std::vector<int> to_host_vector() const;
  };

}
}