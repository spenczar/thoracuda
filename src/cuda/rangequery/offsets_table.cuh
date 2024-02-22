#pragma once

#include "rangequery/data_handle.cuh"
#include "rangequery/counts_table.cuh"

namespace thoracuda {
  namespace rangequery {
    /** Represents a table of offsets into a NeighborsTable. This is
	an inclusive prefix sum over the counts table.
     */
    struct OffsetsTable {
      int *offsets;
      int n_offsets;

      OffsetsTable();
      /// Construct the offsets table from a pre-computed counts table.
      OffsetsTable(const CountsTable &counts);
      /// Construct the offsets table from the data and range query parameters.
      OffsetsTable(const DataHandle &data, float radius);
      ~OffsetsTable();

      OffsetsTable(const OffsetsTable&) = delete;
      OffsetsTable& operator=(const OffsetsTable&) = delete;

      OffsetsTable(OffsetsTable &&other);
      OffsetsTable& operator=(OffsetsTable &&other);

      /// The number of offsets in the table. This requires a read
      /// from device memory.
      int n_total_results() const;

    private:
      void initialize_from_counts(const CountsTable &counts);
    };
  }
}