#include <catch2/catch_test_macros.hpp>
#include "pointsources.hpp"

using Eigen::Vector3d;
using Eigen::Vector4d;

namespace tc = thoracuda;

TEST_CASE("CartesianPointSources constructor with just string", "[CartesianPointSources]") {
  tc::CartesianPointSources cps = tc::CartesianPointSources("500");
  REQUIRE(cps.size() == 0);
  REQUIRE(cps.obscode == "500");
}

TEST_CASE("CartesianPointSources constructor with capacity", "[CartesianPointSources]") {
  tc::CartesianPointSources cps2 = tc::CartesianPointSources("500", 10);
  REQUIRE(cps2.size() == 0);
  REQUIRE(cps2.obscode == "500");
}

TEST_CASE("CartesianPointSources constructor with empty vectors", "[CartesianPointSources]") {
  tc::CartesianPointSources cps3 = tc::CartesianPointSources("500", std::vector<double>(), std::vector<double>(),
																														 std::vector<double>(), std::vector<double>());

  REQUIRE(cps3.size() == 0);
  REQUIRE(cps3.obscode == "500");
}

TEST_CASE("CartesianPointSources constructor with populated vectors", "[CartesianPointSources]") {
  std::vector<double> x = {1.0, 2.0, 3.0};
  std::vector<double> y = {4.0, 5.0, 6.0};
  std::vector<double> z = {7.0, 8.0, 9.0};
  std::vector<double> t = {10.0, 11.0, 12.0};

  tc::CartesianPointSources cps5 = tc::CartesianPointSources("500", x, y, z, t);

  REQUIRE(cps5.size() == 3);
  REQUIRE(cps5.obscode == "500");
  REQUIRE(cps5.nth_pos(0).isApprox(Vector3d(1.0, 4.0, 7.0)));
}

TEST_CASE("CartesianPointSources add", "[CartesianPointSources]") {
  tc::CartesianPointSources cps = tc::CartesianPointSources("500");

  SECTION("Add one point") {
    cps.add(1.0, 2.0, 3.0, 4.0);
    REQUIRE(cps.size() == 1);
    REQUIRE(cps.obscode == "500");
    REQUIRE(cps.nth_pos(0).isApprox(Vector3d(1.0, 2.0, 3.0)));
  }

  SECTION("Add two points") {
    cps.add(1.0, 2.0, 3.0, 4.0);
    cps.add(5.0, 6.0, 7.0, 8.0);
    REQUIRE(cps.size() == 2);
    REQUIRE(cps.obscode == "500");
    REQUIRE(cps.nth_pos(0).isApprox(Vector3d(1.0, 2.0, 3.0)));
    REQUIRE(cps.nth_pos(1).isApprox(Vector3d(5.0, 6.0, 7.0)));
  }
}

TEST_CASE("CartesianPointSources nth", "[CartesianPointSources]") {
  tc::CartesianPointSources cps = tc::CartesianPointSources("500");

  cps.add(1.0, 2.0, 3.0, 4.0);
  cps.add(5.0, 6.0, 7.0, 8.0);

  REQUIRE(cps.nth(0).isApprox(Vector4d(1.0, 2.0, 3.0, 4.0)));
  REQUIRE(cps.nth(1).isApprox(Vector4d(5.0, 6.0, 7.0, 8.0)));
  REQUIRE_THROWS(cps.nth(2));
  REQUIRE_THROWS(cps.nth(-1));
}

TEST_CASE("CartesianPointSources nth_pos", "[CartesianPointSources]") {
  tc::CartesianPointSources cps = tc::CartesianPointSources("500");

  cps.add(1.0, 2.0, 3.0, 4.0);
  cps.add(5.0, 6.0, 7.0, 8.0);

  REQUIRE(cps.nth_pos(0).isApprox(Vector3d(1.0, 2.0, 3.0)));
  REQUIRE(cps.nth_pos(1).isApprox(Vector3d(5.0, 6.0, 7.0)));
  REQUIRE_THROWS(cps.nth_pos(2));
  REQUIRE_THROWS(cps.nth_pos(-1));
}
