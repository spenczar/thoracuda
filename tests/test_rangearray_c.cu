#include <iostream>
#include <random>
#include <cmath>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>


#include "rangearray_c.h"

using Catch::Approx;

TEST_CASE("xyvec_parallel", "") {
  int result;
  int n = 10000;

  struct XYPair *xys = (struct XYPair*)(malloc(n * sizeof(struct XYPair)));
  struct XYBounds actual_bounds = {
    .xmin = INFINITY,
    .xmax = -INFINITY,
    .ymin = INFINITY,
    .ymax = -INFINITY,
  };
  for (int i = 0; i < n; i++) {
    xys[i].x = rand() % 100000;
    xys[i].y = rand() % 100000;
    if(xys[i].x < actual_bounds.xmin) {
      actual_bounds.xmin = xys[i].x;
    }
    if(xys[i].x > actual_bounds.xmax) {
      actual_bounds.xmax = xys[i].x;
    }
    if(xys[i].y < actual_bounds.ymin) {
      actual_bounds.ymin = xys[i].y;
    }
    if(xys[i].y > actual_bounds.ymax) {
      actual_bounds.ymax = xys[i].y;
    }
  }
  struct XYBounds have_bounds;

  result = xy_bounds_parallel(xys, n, &have_bounds);
  REQUIRE(result == 0);

  REQUIRE(have_bounds.xmin == actual_bounds.xmin);
  REQUIRE(have_bounds.xmax == actual_bounds.xmax);
  REQUIRE(have_bounds.ymin == actual_bounds.ymin);
  REQUIRE(have_bounds.ymax == actual_bounds.ymax);

  BENCHMARK("xy_bounds_parallel") {
    return xy_bounds_parallel(xys, n, &have_bounds);
  };

  have_bounds = xy_bounds_serial(xys, n);
  REQUIRE(have_bounds.xmin == actual_bounds.xmin);
  REQUIRE(have_bounds.xmax == actual_bounds.xmax);
  REQUIRE(have_bounds.ymin == actual_bounds.ymin);
  REQUIRE(have_bounds.ymax == actual_bounds.ymax);

  BENCHMARK("xy_bounds_serial") {
    return xy_bounds_serial(xys, n);
  };
}

