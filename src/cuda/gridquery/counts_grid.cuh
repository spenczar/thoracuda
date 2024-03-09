#pragma once

#include <vector>

#include "pair.h"
#include "pairminmax.h"
#include "rangequery/data_handle.cuh"
#include "gridquery/sorted_quantized_data.cuh"

using thoracuda::rangequery::DataHandle;
using thoracuda::gridquery::SortedQuantizedData;

namespace thoracuda {
namespace gridquery {
  struct CountsGrid {
    /// d_counts is (n_cells * n_cells) 2D array of the count of how
    /// many data points are in the (i, j)th cell.
    int *d_counts;
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

    void range_counts(XYPair *d_pairs_in, int n, float radius, int *d_counts_out);
  };

  __global__ void map_cell_counts(int *counts, int n_cells, int *quantized, int *run_offsets, int *run_lengths, int num_runs);
}
}