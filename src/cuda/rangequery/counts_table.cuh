#pragma once

#include "rangequery/row.cuh"
#include "rangequery/data_handle.cuh"

using thoracuda::rangequery::DataHandle;

namespace thoracuda {
  namespace rangequery {
    // Count of how many neighbors each point has.
    struct CountsTable {
      // Pointer to an array of counts, stored on the device. In
      // identical order to the input data handle.
      int *counts;
      int n_counts;

      CountsTable();
      /// Compute the counts table for the given data handle.
      CountsTable(const DataHandle &data, float radius, int threads_per_block=512);
      /// Free the memory associated with the counts table.
      ~CountsTable();

      CountsTable(const CountsTable&) = delete;
      CountsTable& operator=(const CountsTable&) = delete;

      CountsTable(CountsTable &&other);
      CountsTable& operator=(CountsTable &&other);
    };

    /// Low-level CUDA kernel to count how many points are within a
    /// certain distance of each point.
    ///
    /// Two points at the same time are considered infinitely far
    /// apart.
    __global__ void count_within_distance(Row *rows, int n, float sq_distance, int *result);
  }
}
  