#pragma once

#include <vector>

#include "rangequery/row.cuh"
#include "rangequery/data_handle.cuh"
#include "rangequery/offsets_table.cuh"

using thoracuda::rangequery::OffsetsTable;
using thoracuda::rangequery::DataHandle;
using thoracuda::rangequery::Row;

namespace thoracuda {
  namespace rangequery {

    /** Represents a table of range query results: for each point, the
	IDs of its neighbors. This is expected to be used in
	conjunction with an OffsetsTable which holds indexes into the
	*neighbors array.
     */
    struct NeighborsTable {
      int *neighbors;
      int n_neighbors;

      NeighborsTable();
      NeighborsTable(const DataHandle &data, float radius, const OffsetsTable &offsets);
      ~NeighborsTable();

      NeighborsTable(const NeighborsTable&) = delete;
      NeighborsTable& operator=(const NeighborsTable&) = delete;

      NeighborsTable(NeighborsTable &&other);
      NeighborsTable& operator=(NeighborsTable &&other);

      /// Extract the neighbors as a vector of vectors into host memory.
      std::vector<std::vector<int>> get_neighbors(const OffsetsTable &offsets) const;
    };
    __global__ void range_query(Row *rows, int n, float sq_distance, int *offsets, int *result);    
  }
}