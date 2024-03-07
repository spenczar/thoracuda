#pragma once

#include <vector>

#include "gridquery/quantized_data.cuh"

using thoracuda::gridquery::QuantizedData;

namespace thoracuda {
namespace gridquery {

  struct SortedQuantizedData {
    int *sorted_quantized;
    int n;

    SortedQuantizedData(const QuantizedData &qd);

    ~SortedQuantizedData();

    SortedQuantizedData(const SortedQuantizedData &other) = delete;
    SortedQuantizedData &operator=(const SortedQuantizedData &other) = delete;

    SortedQuantizedData(SortedQuantizedData &&other);
    SortedQuantizedData &operator=(SortedQuantizedData &&other);

    std::vector<int> to_host_vector() const;
  };

}
}