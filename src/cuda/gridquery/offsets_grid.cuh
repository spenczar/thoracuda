#pragma once

#include "gridquery/counts_grid.cuh"

using thoracuda::gridquery::CountsGrid;

namespace thoracuda {
namespace gridquery {
  struct OffsetsGrid {
    int *d_offsets;
    int n_cells;

    OffsetsGrid(const CountsGrid &counts_grid);
    ~OffsetsGrid();

    OffsetsGrid(const OffsetsGrid &other) = delete;
    OffsetsGrid &operator=(const OffsetsGrid &other) = delete;

    OffsetsGrid(OffsetsGrid &&other);
    OffsetsGrid &operator=(OffsetsGrid &&other);
  };
}
}