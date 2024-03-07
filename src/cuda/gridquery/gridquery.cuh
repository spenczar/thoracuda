#pragma once

#include <vector>

#include "pair.h"
#include "pairminmax.h"
#include "rangequery/data_handle.cuh"
#include "rangequery/row.cuh"

using thoracuda::rangequery::DataHandle;

namespace thoracuda {
namespace gridquery {

  struct QuantizedData {
    int *quantized;
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

  __global__ void copy_rows_to_xypairs(thoracuda::rangequery::Row *rows, int n, struct XYPair *pairs);
  __global__ void quantize_data(struct XYPair *pairs, int n, struct XYBounds bounds, int n_cells, int *quantized);

  struct CountsGrid {
    /// counts is (n_cells * n_cells) 2D array of the count of how
    /// many data points are in the (i, j)th cell.
    int *counts;
    int n_cells;
    struct XYBounds bounds;

    CountsGrid(int n_cells, struct XYBounds bounds);
    CountsGrid(const DataHandle &dh, const SortedQuantizedData &sqd, int n_cells, struct XYBounds bounds);
    ~CountsGrid();

    CountsGrid(const CountsGrid &other) = delete;
    CountsGrid &operator=(const CountsGrid &other) = delete;

    CountsGrid(CountsGrid &&other);
    CountsGrid &operator=(CountsGrid &&other);

    /// Returns the grid as a 2D, row-major vector
    std::vector<std::vector<int>> to_host_vector() const;
  };

  __global__ void map_cell_counts(int *counts, int n_cells, int *quantized, int *run_offsets, int *run_lengths, int num_runs);    
}

}