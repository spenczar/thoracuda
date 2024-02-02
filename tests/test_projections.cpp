#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <random>
#include <Eigen/Dense>

#include "projections.hpp"

using Eigen::Matrix3d;
using Eigen::Vector3d;

using Catch::Approx;

namespace tcp = thoracuda::projections;

// A center position and velocity which were used in moeyensj/thor to
// generate "golden" test data which we're trying to replicate.
static Eigen::Vector3d center_pos = {2.32545784897911, -0.459940068868785, 0.0788698905258432};
static Eigen::Vector3d center_vel = {0.00257146073153728, 0.011315544836752, 0.00041171196985311};

TEST_CASE("R1 matrix for perfect Z-axis alignment", "[r1_matrix]") {
  Eigen::Vector3d center_pos = {1.0, 0.0, 0.0};
  Eigen::Vector3d center_vel = {0.0, 1.0, 0.0};

  Eigen::Matrix3d r1 = tcp::r1_matrix(center_pos, center_vel);

  Eigen::Matrix3d r1_expected = Eigen::Matrix3d::Identity();
  REQUIRE(r1.isApprox(r1_expected));
}

TEST_CASE("R1 matrix golden test", "[r1_matrix]") {
  Eigen::Matrix3d r1 = tcp::r1_matrix(center_pos, center_vel);

  Eigen::Matrix3d r1_expected;
  r1_expected << 9.992273655465606e-01, -5.389407578454692e-04, 3.929861938719766e-02, -5.389407578454692e-04,
      9.996240691323380e-01, 2.741222271480839e-02, -3.929861938719766e-02, -2.741222271480839e-02,
      9.988514346788986e-01;

  REQUIRE(r1.isApprox(r1_expected));
}

TEST_CASE("R2 matrix golden test", "[r2_matrix]") {
  Eigen::Matrix3d r1 = tcp::r1_matrix(center_pos, center_vel);
  Eigen::Matrix3d r2 = tcp::r2_matrix(r1, center_pos);

  Eigen::Matrix3d r2_expected;
  r2_expected << 0.981107616210045, -0.193462775268636, 0.0, 0.193462775268636, 0.981107616210045, 0.0, 0.0, 0.0, 1.0;

  REQUIRE(r2.isApprox(r2_expected));
}

TEST_CASE("gnomonic rotation matrix", "[gnomonic_rotation_matrix]") {
  Eigen::Matrix3d m = tcp::gnomonic_rotation_matrix(center_pos, center_vel);

  Eigen::Matrix3d m_expected;
  m_expected << 0.980453843637948, -0.193918805521877, 0.033252930104631, 0.192784540380797, 0.980634522597895,
      0.034497160453618, -0.039298619387198, -0.027412222714808, 0.998851434678899;

  REQUIRE(m.isApprox(m_expected));
}

TEST_CASE("gnomonic projections consistent with moeyensj/thor", "[gnomonic_projection]") {
  thoracuda::CartesianPointSources cps("W84");
  cps.add(2.32724566583692, -0.449382385055792, 0.0866176471970003, 56537.2416032334);
  cps.add(2.32728038367672, -0.44938629368127, 0.0856592596632065, 56537.2416032334);
  cps.add(2.32729293752894, -0.449243941523289, 0.0860639174895683, 56537.2416032334);

  Eigen::Vector3d center_pos = {2.32545784897911, -0.459940068868785, 0.0788698905258432};
  Eigen::Vector3d center_vel = {0.00257146073153728, 0.011315544836752, 0.00041171196985311};

  thoracuda::GnomonicPointSources gps = tcp::gnomonic_projection(cps, center_pos, center_vel);

  REQUIRE(gps.size() == 3);

  Eigen::Vector3d p0 = gps.nth(0);
  Eigen::Vector3d p1 = gps.nth(1);
  Eigen::Vector3d p2 = gps.nth(2);

  Eigen::Vector3d p0_expected = {0.26488865499153, 0.178261157983677, 56537.2416032334};
  Eigen::Vector3d p1_expected = {0.264158742296632, 0.155105149861758, 56537.2416032334};
  Eigen::Vector3d p2_expected = {0.267926881709716, 0.16476328675463, 56537.2416032334};

  REQUIRE(p0.isApprox(p0_expected));
  REQUIRE(p1.isApprox(p1_expected));
  REQUIRE(p2.isApprox(p2_expected));

  BENCHMARK("gnomonic projection of 3 points") { return tcp::gnomonic_projection(cps, center_pos, center_vel); };

  // Generate ranndom x, y, z within a box centered on the center_pos,
  // ranging from -0.8 to 1.2 times the center_pos.
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-0.8, 1.2);
  thoracuda::CartesianPointSources cps_100k("W84");
  for (int i = 0; i < 100'000; i++) {
    double x = dis(gen) * center_pos(0);
    double y = dis(gen) * center_pos(1);
    double z = dis(gen) * center_pos(2);
    cps_100k.add(x, y, z, 56537.2416032334);
  }
  BENCHMARK("gnomonic projection of 100,000 points") {
    return tcp::gnomonic_projection(cps_100k, center_pos, center_vel);
  };
}
