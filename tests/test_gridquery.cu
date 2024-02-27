#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "rangequery/data_handle.cuh"
#include "gridquery/gridquery.cuh"

using thoracuda::rangequery::DataHandle;
using thoracuda::gridquery::QuantizedData;

TEST_CASE("grid query quantization", "") {
  std::vector<float> x = {0, 0.1, 0.2, 1.0, 0.9, 0.0};
  std::vector<float> y = {0, 0.1, 0.5, 0.1, -0.5, 0.2};
  std::vector<float> t = {0, 0, 0, 0, 0, 0};
  std::vector<int> ids = {0, 1, 2, 3, 4, 5};

  DataHandle data(x, y, t, ids);

  QuantizedData qd(data, 10);

  std::vector<int2> result = qd.to_host_vector();

  std::vector<int2> expected = {{0, 5}, {1, 6}, {2, 10}, {10, 6}, {9, 0}, {0, 7}};

  REQUIRE(result.size() == expected.size());
  for (size_t i = 0; i < result.size(); i++) {
    REQUIRE(result[i].x == expected[i].x);
    REQUIRE(result[i].y == expected[i].y);
  }
}