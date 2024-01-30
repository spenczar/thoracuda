#include <iostream>
#
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "rangearray.h"

using Catch::Approx;

TEST_CASE("grid calculations", "[cuda]") {
  GridCoordKernel kernel(0.0, 10.0, 0.0, 10.0, 10);
  short2 pair = kernel(make_float2(0, 0));

  REQUIRE(pair.x == 0);
  REQUIRE(pair.y == 0);

  pair = kernel(make_float2(9, 9));

  REQUIRE(pair.x == 9);
  REQUIRE(pair.y == 9);

  kernel = GridCoordKernel(0.0, 10.0, 0.0, 10.0, 5);
  pair = kernel(make_float2(0, 0));

  REQUIRE(pair.x == 0);
  REQUIRE(pair.y == 0);

  pair = kernel(make_float2(9, 9));

  REQUIRE(pair.x == 4);
  REQUIRE(pair.y == 4);

  kernel = GridCoordKernel(-10.0, 10.0, -10.0, 10.0, 250);

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

TEST_CASE("building a grid coord map", "[cuda]") {
  thrust::host_vector<float2> xy(3);
  xy[0] = make_float2(1.0, 4.0);
  xy[1] = make_float2(2.0, 5.0);
  xy[2] = make_float2(3.0, 6.0);

  struct FindPairConfig config;
  config.min_x = 0.0;
  config.max_x = 10.0;
  config.min_y = 0.0;
  config.max_y = 10.0;
  config.grid_n_cells_1d = 10;
  thrust::host_vector<short2> grid_coords = build_grid_coord_map(xy, config);

  REQUIRE(grid_coords.size() == 3);
  REQUIRE(grid_coords[0].x == 1);
  REQUIRE(grid_coords[0].y == 4);
  REQUIRE(grid_coords[1].x == 2);
  REQUIRE(grid_coords[1].y == 5);
}

TEST_CASE("sort by grid cell", "[cuda]") {
  thrust::host_vector<float2> xy_h(3);
  xy_h[0] = make_float2(1.0, 4.0);
  xy_h[1] = make_float2(2.0, 5.0);
  xy_h[2] = make_float2(3.0, 6.0);
  
  thrust::host_vector<short2> grid_coords_h(3);
  grid_coords_h[0] = make_short2(1, 4);
  grid_coords_h[1] = make_short2(2, 5);
  grid_coords_h[2] = make_short2(1, 6);

  thrust::host_vector<int> indices_h(3);
  indices_h[0] = 0;
  indices_h[1] = 1;
  indices_h[2] = 2;

  thrust::device_vector<float2> xy = xy_h;
  thrust::device_vector<short2> grid_coords = grid_coords_h;
  thrust::device_vector<int> indices = indices_h;

  sort_by_grid_cell(grid_coords, xy, indices);

  xy_h = xy;
  grid_coords_h = grid_coords;
  indices_h = indices;

  REQUIRE(indices_h[0] == 0);
  REQUIRE(indices_h[1] == 2);
  REQUIRE(indices_h[2] == 1);

  REQUIRE(grid_coords_h[0].x == 1);
  REQUIRE(grid_coords_h[0].y == 4);
  REQUIRE(grid_coords_h[1].x == 1);
  REQUIRE(grid_coords_h[1].y == 6);
  REQUIRE(grid_coords_h[2].x == 2);
  REQUIRE(grid_coords_h[2].y == 5);
}


TEST_CASE("build dense matrix", "[cuda]") {

  thrust::host_vector<short2> grid_coords_h(10);
  grid_coords_h[0] = make_short2(0, 0);
  grid_coords_h[1] = make_short2(0, 0);
  grid_coords_h[2] = make_short2(0, 0);
  grid_coords_h[3] = make_short2(1, 1);
  grid_coords_h[4] = make_short2(1, 2);
  grid_coords_h[5] = make_short2(2, 0);
  grid_coords_h[6] = make_short2(2, 0);
  grid_coords_h[7] = make_short2(2, 0);
  grid_coords_h[8] = make_short2(2, 0);
  grid_coords_h[9] = make_short2(3, 2);

  int grid_n_cells_1d = 4;

  thrust::host_vector<int2> expected_h(16);
  /* Cells hold (index, count) pairs.
   * Expected output:
   * (0, 3), (0, 0), (0, 0), (0, 0)
   * (0, 0), (3, 4), (4, 5), (0, 0)
   * (5, 9), (0, 0), (0, 0), (0, 0)
   * (0, 0), (0, 0), (9, 10), (0, 0)
   */
  expected_h[0] = make_int2(0, 3);
  expected_h[1] = make_int2(0, 0);
  expected_h[2] = make_int2(0, 0);
  expected_h[3] = make_int2(0, 0);

  expected_h[4] = make_int2(0, 0);
  expected_h[5] = make_int2(3, 4);
  expected_h[6] = make_int2(4, 5);
  expected_h[7] = make_int2(0, 0);

  expected_h[8] = make_int2(5, 9);
  expected_h[9] = make_int2(0, 0);
  expected_h[10] = make_int2(0, 0);
  expected_h[11] = make_int2(0, 0);

  expected_h[12] = make_int2(0, 0);
  expected_h[13] = make_int2(0, 0);
  expected_h[14] = make_int2(9, 10);
  expected_h[15] = make_int2(0, 0);
  thrust::device_vector<short2> grid_coords_d = grid_coords_h;
  thrust::device_vector<int2> actual_d = build_dense_matrix(grid_coords_d, grid_n_cells_1d);

  thrust::host_vector<int2> actual_h = actual_d;

  REQUIRE(actual_h.size() == 16);
  for (int i = 0; i < 16; i++) {
    REQUIRE(actual_h[i].x == expected_h[i].x);
    REQUIRE(actual_h[i].y == expected_h[i].y);
  }
}