#include <vector>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "gnomonic_point_sources.hpp"
#include "range_brute.h"

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

