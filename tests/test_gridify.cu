#include <catch2/catch_test_macros.hpp>

#include "gridify.h"
#include "pair.h"

TEST_CASE("gridify_serial", "") {
  int N = 1000;
  struct XYPair xys[N];
  double t[N];
  for (int i = 0; i < N; i++) {
    xys[i].x = i;
    xys[i].y = i;
    t[i] = i / 100.0;
  }

  struct Grid grid;
  
  int result = gridify_points_serial(xys, t, N, &grid);

  REQUIRE(result == 0);
  // Should be 10 unique times, so there should be 64 * 64 * 10 = 40960 cells
  REQUIRE(grid.address_map_n == 40960);

  // Check that the grid is correct
  struct CellPosition cp;
  struct XYPair xy = {.x = 10, .y = 10};
  struct XYBounds bounds = {.xmin = 0, .xmax = N, .ymin=0, .ymax=N};
  cp = xy_to_cell(
		  xy,
		  bounds,
		  0,
		  0);
  REQUIRE(cp.x == 0);
  REQUIRE(cp.y == 0);
  REQUIRE(cp.t == 0);

  struct IndexPair ip = grid_query(cp, &grid);
  REQUIRE(ip.start == 0);
  REQUIRE(ip.end == 15);
}

