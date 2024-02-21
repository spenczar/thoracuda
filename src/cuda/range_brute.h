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

    // Count of how many neighbors each point has.
    struct CountsTable {
      int *counts;
      int n_counts;

      CountsTable();
      CountsTable(const DataHandle &data, const RangeQueryParams &params);
      ~CountsTable();

      CountsTable(const CountsTable&) = delete;
      CountsTable& operator=(const CountsTable&) = delete;

      CountsTable(CountsTable &&other);
      CountsTable& operator=(CountsTable &&other);
    };


    struct OffsetsTable {
      int *offsets;
      int n_offsets;

      OffsetsTable();
      // Construct the offsets table from a pre-computed counts table.
      OffsetsTable(const CountsTable &counts);
      // Construct the offsets table from the data and range query parameters.
      OffsetsTable(const DataHandle &data, const RangeQueryParams &params);
      ~OffsetsTable();

      OffsetsTable(const OffsetsTable&) = delete;
      OffsetsTable& operator=(const OffsetsTable&) = delete;

      OffsetsTable(OffsetsTable &&other);
      OffsetsTable& operator=(OffsetsTable &&other);
      
      int n_total_results() const;

    private:
      void initialize_from_counts(const CountsTable &counts);
    };

    struct NeighborsTable {
      int *neighbors;
      int n_neighbors;

      NeighborsTable();
      NeighborsTable(const DataHandle &data, const RangeQueryParams &params, const OffsetsTable &offsets);
      ~NeighborsTable();

      NeighborsTable(const NeighborsTable&) = delete;
      NeighborsTable& operator=(const NeighborsTable&) = delete;

      NeighborsTable(NeighborsTable &&other);
      NeighborsTable& operator=(NeighborsTable &&other);

      std::vector<std::vector<int>> get_neighbors() const;
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

    // Returns the ID of the cluster for each point.
    std::vector<int> DBSCAN(float epsilon, int min_points, const Inputs &inputs);

    __global__ void identify_core_points(int *counts, int n, int min_points, int *cluster_ids);
    __global__ void collapse_clusters(int *cluster_ids, int n, int *offsets, int *counts, int *neighbors);
    __global__ void associate_points_to_core_points(int *counts, int n, int *offsets, int *neighbors, int *cluster_ids, int min_points);
    __device__ int min_neighbor_cluster_id(int i, int *cluster_ids, int *offsets, int *counts, int *neighbors);    
  }
}
