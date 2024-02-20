#pragma once

#include <vector>

namespace thoracuda {
  namespace rangequery {
    struct Inputs {
      // Host-side data
      std::vector<float> x;
      std::vector<float> y;
      std::vector<float> t;
      std::vector<int> id;
    };

    class DataHandle {
    public:
      // Device-side data
      float *x;
      float *y;
      float *t;
      int *id;

      // Shared for all above arrays
      int n;
      DataHandle(const Inputs &inputs);
      ~DataHandle();

      DataHandle(const DataHandle&) = delete;
      DataHandle& operator=(const DataHandle&) = delete;

      DataHandle(DataHandle &&other);
      DataHandle& operator=(DataHandle &&other);
    };

    enum RangeQueryMetric {
      EUCLIDEAN,
      MANHATTAN
    };

    struct RangeQueryParams {
      RangeQueryMetric metric;
      float distance;
    };


    struct OffsetsTable {
      int *offsets;
      int n_offsets;

      OffsetsTable();
      OffsetsTable(const DataHandle &data, const RangeQueryParams &params);
      ~OffsetsTable();

      OffsetsTable(const OffsetsTable&) = delete;
      OffsetsTable& operator=(const OffsetsTable&) = delete;

      OffsetsTable(OffsetsTable &&other);
      OffsetsTable& operator=(OffsetsTable &&other);
      
      int n_total_results();
    };
    
    struct Results {
      int *results;
      int n_results;

      OffsetsTable offsets;
      std::vector<std::vector<int>> get_results();

      Results(int *results, int n_results, OffsetsTable offsets);
      ~Results();
      
      Results(const Results&) = delete;
      Results& operator=(const Results&) = delete;

      Results(Results &&other);
      Results& operator=(Results &&other);
    };

    
    Results range_query(const Inputs &inputs, const RangeQueryParams &params);

    __global__ void count_within_euclidean_distance(float *x, float *y, float *t, int n, float sq_distance, int *result);
    __global__ void count_within_manhattan_distance(float *x, float *y, float *t, int n, float distance, int *result);
    __global__ void range_query_euclidean(float *x, float *y, float *t, int *id, int n, float sq_distance, int *offsets, int *result);
    __global__ void range_query_manhattan(float *x, float *y, float *t, int *id, int n, float distance, int *offsets, int *result);
  }
}
