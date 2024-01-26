#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "rangearray.h"

using Catch::Approx;

TEST_CASE("grid calculations", "[cuda]") {
  GridCoordKernel kernel(0, 10, 10, 0, 10, 10);
  struct IntPair pair = kernel(0, 0);

  REQUIRE(pair.x == 0);
  REQUIRE(pair.y == 0);

  pair = kernel(9, 9);

  REQUIRE(pair.x == 9);
  REQUIRE(pair.y == 9);

  kernel = GridCoordKernel(0, 10, 5, 0, 10, 5);
  pair = kernel(0, 0);

  REQUIRE(pair.x == 0);
  REQUIRE(pair.y == 0);

  pair = kernel(9, 9);

  REQUIRE(pair.x == 4);
  REQUIRE(pair.y == 4);

  kernel = GridCoordKernel(-10, 10, 250, -10, 10, 250);

  pair = kernel(-10, -10);

  REQUIRE(pair.x == 0);
  REQUIRE(pair.y == 0);

  pair = kernel(0, 0);

  REQUIRE(pair.x == 125);
  REQUIRE(pair.y == 125);

  pair = kernel(0.001, 0.001);

  REQUIRE(pair.x == 125);
  REQUIRE(pair.y == 125);
}

TEST_CASE("linking to cuda", "[cuda]") {
  std::vector<float> x = {1.0, 2.0, 3.0};
  std::vector<float> y = {4.0, 5.0, 6.0};
  std::vector<int> id = {0, 1, 2};
  
  Exposure e = Exposure(x, y, id, 0.0);

  struct FindPairConfig config;
  config.min_x = 0.0;
  config.max_x = 10.0;
  config.min_y = 0.0;
  config.max_y = 10.0;
  config.grid_dim_x = 10;
  config.grid_dim_y = 10;

  thrust::device_vector<struct IntPair> device_grid_coords = build_grid_coord_map(e, config);

  thrust::host_vector<struct IntPair> grid_coords = device_grid_coords;

  REQUIRE(grid_coords.size() == 3);
  REQUIRE(grid_coords[0].x == 1);
  REQUIRE(grid_coords[0].y == 4);
  REQUIRE(grid_coords[1].x == 2);
  REQUIRE(grid_coords[1].y == 5);
}
