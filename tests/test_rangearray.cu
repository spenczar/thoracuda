#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "rangearray.h"

using Catch::Approx;

TEST_CASE("grid calculations", "[cuda]") {
  GridCoordKernel kernel(0, 10, 10, 0, 10, 10);
  short2 pair = kernel(make_float2(0, 0));

  REQUIRE(pair.x == 0);
  REQUIRE(pair.y == 0);

  pair = kernel(make_float2(9, 9));

  REQUIRE(pair.x == 9);
  REQUIRE(pair.y == 9);

  kernel = GridCoordKernel(0, 10, 5, 0, 10, 5);
  pair = kernel(make_float2(0, 0));

  REQUIRE(pair.x == 0);
  REQUIRE(pair.y == 0);

  pair = kernel(make_float2(9, 9));

  REQUIRE(pair.x == 4);
  REQUIRE(pair.y == 4);

  kernel = GridCoordKernel(-10, 10, 250, -10, 10, 250);

  pair = kernel(make_float2(-10, -10));

  REQUIRE(pair.x == 0);
  REQUIRE(pair.y == 0);

  pair = kernel(make_float2(0, 0));

  REQUIRE(pair.x == 125);
  REQUIRE(pair.y == 125);

  pair = kernel(make_float2(0.001, 0.001));

  REQUIRE(pair.x == 125);
  REQUIRE(pair.y == 125);
}

TEST_CASE("linking to cuda", "[cuda]") {
  thrust::host_vector<float2> xy(3);
  xy[0] = make_float2(1.0, 4.0);
  xy[1] = make_float2(2.0, 5.0);
  xy[2] = make_float2(3.0, 6.0);

  struct FindPairConfig config;
  config.min_x = 0.0;
  config.max_x = 10.0;
  config.min_y = 0.0;
  config.max_y = 10.0;
  config.grid_dim_x = 10;
  config.grid_dim_y = 10;
  thrust::host_vector<short2> grid_coords = build_grid_coord_map(xy, config);

  REQUIRE(grid_coords.size() == 3);
  REQUIRE(grid_coords[0].x == 1);
  REQUIRE(grid_coords[0].y == 4);
  REQUIRE(grid_coords[1].x == 2);
  REQUIRE(grid_coords[1].y == 5);
}
