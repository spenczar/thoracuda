#include <vector>

#include <catch2/catch_test_macros.hpp>


#include "rangequery/data_handle.cuh"
#include "gridquery/quantized_data.cuh"
#include "gridquery/sorted_quantized_data.cuh"
#include "gridquery/counts_grid.cuh"
#include "pairminmax.h"

using thoracuda::rangequery::DataHandle;
using thoracuda::gridquery::QuantizedData;
using thoracuda::gridquery::SortedQuantizedData;
using thoracuda::gridquery::CountsGrid;

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

  REQUIRE(qd.bounds.xmin == 0.0f);
  REQUIRE(qd.bounds.xmax == 1.0f);
  REQUIRE(qd.bounds.ymin == -0.5f);
  REQUIRE(qd.bounds.ymax == 0.5f);
}

TEST_CASE("grid query counting", "") {
  std::vector<float> x = {0, 0.1, 0.1, 0.2, 1.0, 0.9, 0.0};
  std::vector<float> y = {-0.5, -0.4, -0.4, 0.5, 0.6, -0.5, 0.2};
  std::vector<float> t = {0, 0, 0, 0, 0, 0, 0};
  std::vector<int> ids = {0, 1, 2, 3, 4, 5, 6};

  DataHandle data(x, y, t, ids);

  int n_cells = 10;
  QuantizedData qd(data, n_cells);
  SortedQuantizedData sqd(qd, data);

  struct XYBounds bounds = {
    0.0f, 1.0f, -0.5f, 0.5f
  };

  CountsGrid cg(data, sqd, n_cells, bounds);

  std::vector<std::vector<int>> result = cg.to_host_vector();

  std::vector<std::vector<int>> expected = {
    {1, 0, 0, 0, 0, 0, 1, 0, 0, 0},
    {2, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},    
    {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},    
  };

  REQUIRE(result.size() == expected.size());
  for (size_t i = 0; i < result.size(); i++) {
    REQUIRE(result[i].size() == expected[i].size());
    for (size_t j = 0; j < result[i].size(); j++) {
      INFO("i: " << i << " j: " << j);
      REQUIRE(result[i][j] == expected[i][j]);
    }
  }
}