#include <random>
#include <map>
#include <vector>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cuda_profiler_api.h>

#define EIGEN_NO_CUDA 1
#include "gnomonic_point_sources.hpp"
#include "testutils.hpp"

#include "rangequery/data_handle.cuh"
#include "rangequery/offsets_table.cuh"
#include "rangequery/neighbors_table.cuh"

using Catch::Approx;
using thoracuda::rangequery::DataHandle;
using thoracuda::rangequery::OffsetsTable;
using thoracuda::rangequery::NeighborsTable;

TEST_CASE("range_query", "") {

  std::vector<float> x = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> y = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> t = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int> ids = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  DataHandle data(x, y, t, ids);

  float radius = 1.5;

  OffsetsTable offsets(data, radius);
  NeighborsTable neighbors(data, radius, offsets);

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

  std::vector<std::vector<int>> actual = neighbors.get_neighbors(offsets);

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

  DataHandle data(x, y, t, ids);

  float radius = 0.03;


  BENCHMARK("range_query_thor_data") {
    OffsetsTable offsets(data, radius);
    NeighborsTable neighbors(data, radius, offsets);
    return neighbors.get_neighbors(offsets);
  };
}
