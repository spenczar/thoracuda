#include <catch2/catch_test_macros.hpp>
#include "projections.hpp"

using Eigen::Matrix3d;
using Eigen::Vector3d;

namespace tcp = thoracuda::projections;

TEST_CASE("R1 matrix for perfect Z-axis alignment", "[r1_matrix]") {
  Eigen::Vector3d center_pos = {1.0, 0.0, 0.0};
  Eigen::Vector3d center_vel = {0.0, 1.0, 0.0};

  Eigen::Matrix3d r1 = tcp::r1_matrix(center_pos, center_vel);

  Eigen::Matrix3d r1_expected = Eigen::Matrix3d::Identity();
  REQUIRE(r1.isApprox(r1_expected));
}

