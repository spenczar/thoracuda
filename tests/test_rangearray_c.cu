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
  int n = 100;
  
  struct XYPairVector xyvec;
  result = xyvec_init(&xyvec, n);
  REQUIRE(result == 0);

  struct XYVectorBounds actual_bounds = {
    .xmin = INFINITY,
    .xmax = -INFINITY,
    .ymin = INFINITY,
    .ymax = -INFINITY,
  };
  for (int i = 0; i < n; i++) {
    xyvec.xy[i].x = rand() % 100000;
    xyvec.xy[i].y = rand() % 100000;
    if(xyvec.xy[i].x < actual_bounds.xmin) {
      actual_bounds.xmin = xyvec.xy[i].x;
    }
    if(xyvec.xy[i].x > actual_bounds.xmax) {
      actual_bounds.xmax = xyvec.xy[i].x;
    }
    if(xyvec.xy[i].y < actual_bounds.ymin) {
      actual_bounds.ymin = xyvec.xy[i].y;
    }
    if(xyvec.xy[i].y > actual_bounds.ymax) {
      actual_bounds.ymax = xyvec.xy[i].y;
    }
  }
  struct XYVectorBounds have_bounds;

  result = xyvec_bounds_parallel(&xyvec, &have_bounds);
  REQUIRE(result == 0);

  REQUIRE(have_bounds.xmin == actual_bounds.xmin);
  REQUIRE(have_bounds.xmax == actual_bounds.xmax);
  REQUIRE(have_bounds.ymin == actual_bounds.ymin);
  REQUIRE(have_bounds.ymax == actual_bounds.ymax);

  BENCHMARK("xyvec_bounds_parallel") {
    return xyvec_bounds_parallel(&xyvec, &have_bounds);
  };

  std::cout << "serial implementation" << std::endl;
  have_bounds = xyvec_bounds(&xyvec);
  REQUIRE(have_bounds.xmin == actual_bounds.xmin);
  REQUIRE(have_bounds.xmax == actual_bounds.xmax);
  REQUIRE(have_bounds.ymin == actual_bounds.ymin);
  REQUIRE(have_bounds.ymax == actual_bounds.ymax);

  std::cout << "serial implementation done" << std::endl;
  BENCHMARK("xyvec_bounds_serial") {
    return xyvec_bounds(&xyvec);
  };
}

