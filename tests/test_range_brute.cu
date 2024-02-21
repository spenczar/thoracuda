#include <random>
#include <vector>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#define EIGEN_NO_CUDA 1
#include "gnomonic_point_sources.hpp"
#include "range_brute.h"
#include "testutils.hpp"

using Catch::Approx;
using thoracuda::rangequery::Inputs;
using thoracuda::rangequery::DataHandle;
using thoracuda::rangequery::RangeQueryParams;
using thoracuda::rangequery::Results;
using thoracuda::rangequery::range_query;

TEST_CASE("range_query", "") {

  std::vector<float> x = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> y = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> t = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int> ids = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  Inputs inputs = {x, y, t, ids};

  RangeQueryParams params = {
    thoracuda::rangequery::RangeQueryMetric::EUCLIDEAN,
    1.5f,
  };

  Results results = range_query(inputs, params);

  std::vector<std::vector<int>> expected = {
    {1},
    {0, 2},
    {1, 3},
    {2, 4},
    {3, 5},
    {4, 6},
    {5, 7},
    {6, 8},
    {7, 9},
    {8},
  };

  std::vector<std::vector<int>> actual = results.get_results();

  REQUIRE(actual == expected);
}

static std::vector<float> doubles_to_floats(const std::vector<double>& doubles) {
  std::vector<float> floats;
  floats.reserve(doubles.size());
  for (auto d : doubles) {
    floats.push_back(static_cast<float>(d));
  }
  return floats;
}

TEST_CASE("range_query_thor_data", "") {
  auto gps = read_point_data();

  std::vector<float> x = doubles_to_floats(gps.x);
  std::vector<float> y = doubles_to_floats(gps.y);
  std::vector<float> t = doubles_to_floats(gps.t);
  std::vector<int> ids;
  ids.reserve(gps.x.size());
  for (int i = 0; i < gps.x.size(); i++) {
    ids.push_back(i);
  }

  Inputs inputs = {x, y, t, ids};

  RangeQueryParams euc_params = {
    thoracuda::rangequery::RangeQueryMetric::EUCLIDEAN,
    0.03f,
  };


  BENCHMARK("range_query_thor_euclidean") {
    Results results = range_query(inputs, euc_params);
    return results;
  };

  RangeQueryParams man_params = {
    thoracuda::rangequery::RangeQueryMetric::MANHATTAN,
    0.03f,
  };

  BENCHMARK("range_query_thor_manhattan") {
    Results results = range_query(inputs, man_params);
    return results;
  };
  
}

TEST_CASE("dbscan_brute", "") {
  std::vector<float> x = {1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<float> y = {6.0, 6.0, 6.0, 6.0, 7.0, 8.0, 9.0, 10.0};
  std::vector<float> t = {11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0};
  std::vector<int> ids = {0, 1, 2, 3, 4, 5, 6, 7};

  float epsilon = 2.0;
  int min_points = 5;

  Inputs inputs = {x, y, t, ids};

  std::vector<int> clusters = DBSCAN(epsilon, min_points, inputs);

  REQUIRE(clusters.size() == ids.size());
  REQUIRE(clusters[0] == 0);
  REQUIRE(clusters[1] == 0);
  REQUIRE(clusters[2] == 0);
  REQUIRE(clusters[3] == 0);
  REQUIRE(clusters[4] == 0);
  REQUIRE(clusters[5] == 0);
  REQUIRE(clusters[6] == -1);
  REQUIRE(clusters[7] == -1);
  
  // Benchmark with N=10000.
  //
  // True clusters at a few select locations, and then piles of noise.

  int N = 10000;
  epsilon = 0.1;
  float x_min = -10.0;
  float x_max = 10.0;
  float y_min = -10.0;
  float y_max = 10.0;
  std::random_device rd; 
  std::mt19937 gen(rd());

  std::uniform_real_distribution<> x_distr(x_min, x_max);
  std::uniform_real_distribution<> y_distr(y_min, y_max);
  std::uniform_int_distribution<> cluster_size_distr(min_points, 8*min_points);
  int N_cluster = 50;
  int total_cluster_points_generated = 0;
  for (int i = 0; i < N_cluster; i++) {
    float x_center = x_distr(gen);
    float y_center = y_distr(gen);
    int cluster_size = cluster_size_distr(gen);

    std::normal_distribution<> x_dist(x_center, epsilon/2.0);
    std::normal_distribution<> y_dist(y_center, epsilon/2.0);

    for (int j = 0; j < cluster_size; j++) {
      x.push_back(x_dist(gen));
      y.push_back(y_dist(gen));
      t.push_back(j);
      ids.push_back(i);
    }
    total_cluster_points_generated += cluster_size;
  }
  // Bunch of uniform random points
  for (int i = 0; i < N - total_cluster_points_generated; i++) {
    x.push_back(x_distr(gen));
    y.push_back(y_distr(gen));
    t.push_back(i);
    ids.push_back(i);
  }

  inputs = {x, y, t, ids};

  BENCHMARK("dbscan_cuda_brute") {
    std::vector<int> clusters = DBSCAN(epsilon, min_points, inputs);
    return clusters;
  };
};

TEST_CASE("dbscan bench against THOR data", "") {
  auto gps = read_point_data();
  REQUIRE(gps.size() == 11153);
  
  std::vector<float> x = doubles_to_floats(gps.x);
  std::vector<float> y = doubles_to_floats(gps.y);
  std::vector<float> t = doubles_to_floats(gps.t);
  std::vector<int> ids;
  ids.reserve(gps.x.size());
  for (int i = 0; i < gps.x.size(); i++) {
    ids.push_back(i);
  }

  Inputs inputs = {x, y, t, ids};

  float eps = 0.002777777777777778;
  int min_size = 6;

  std::vector<int> clusters = DBSCAN(eps, min_size, inputs);

  BENCHMARK("dbscan_on_thor") {
    std::vector<int> clusters = DBSCAN(eps, min_size, inputs);    
    return clusters;
  };
}
