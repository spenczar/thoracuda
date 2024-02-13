#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "gnomonic_point_sources.hpp"
#include "kdtree.hpp"

using Eigen::Vector3d;
using thoracuda::GnomonicPointSources;

TEST_CASE("kd tree construction", "") {
  std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<double> y = {6.0, 7.0, 8.0, 9.0, 10.0};
  std::vector<double> t = {11.0, 12.0, 13.0, 14.0, 15.0};

  GnomonicPointSources gps = GnomonicPointSources(x, y, t);
  thoracuda::kdtree::KDNode kd = thoracuda::kdtree::build_kdtree(gps, 2);

  REQUIRE(kd.dim == thoracuda::kdtree::SplitDimension::X);
  REQUIRE(kd.split == 3.0);

  REQUIRE(kd.left_leaf != nullptr);
  REQUIRE(kd.left_leaf->points.size() == 2);
  REQUIRE(kd.left_leaf->points.nth(0).isApprox(Vector3d(1.0, 6.0, 11.0)));
  REQUIRE(kd.left_leaf->points.nth(1).isApprox(Vector3d(2.0, 7.0, 12.0)));
  REQUIRE(kd.left_leaf->ids.size() == 2);
  REQUIRE(kd.left_leaf->ids[0] == 0);
  REQUIRE(kd.left_leaf->ids[1] == 1);

  REQUIRE(kd.right_kd != nullptr);
  REQUIRE(kd.right_kd->dim == thoracuda::kdtree::SplitDimension::Y);
  REQUIRE(kd.right_kd->split == 9.0);

  REQUIRE(kd.right_kd->left_leaf != nullptr);
  REQUIRE(kd.right_kd->left_leaf->points.size() == 1);
  REQUIRE(kd.right_kd->left_leaf->points.nth(0).isApprox(Vector3d(3.0, 8.0, 13.0)));
  REQUIRE(kd.right_kd->left_leaf->ids.size() == 1);
  REQUIRE(kd.right_kd->left_leaf->ids[0] == 2);

  REQUIRE(kd.right_kd->right_leaf != nullptr);
  REQUIRE(kd.right_kd->right_leaf->points.size() == 2);
  REQUIRE(kd.right_kd->right_leaf->points.nth(0).isApprox(Vector3d(4.0, 9.0, 14.0)));
  REQUIRE(kd.right_kd->right_leaf->points.nth(1).isApprox(Vector3d(5.0, 10.0, 15.0)));
  REQUIRE(kd.right_kd->right_leaf->ids.size() == 2);
  REQUIRE(kd.right_kd->right_leaf->ids[0] == 3);
  REQUIRE(kd.right_kd->right_leaf->ids[1] == 4);

  SECTION("range queries") {
    thoracuda::kdtree::IDPoints id_points = kd.range_query(2.5, 4.5, 7.5, 9.5);
    REQUIRE(id_points.ids.size() == 2);
    REQUIRE(id_points.ids[0] == 2);
    REQUIRE(id_points.ids[1] == 3);
    REQUIRE(id_points.points.size() == 2);
    REQUIRE(id_points.points.nth(0).isApprox(Vector3d(3.0, 8.0, 13.0)));
    REQUIRE(id_points.points.nth(1).isApprox(Vector3d(4.0, 9.0, 14.0)));

    id_points = kd.range_query(2.5, 4.5, 6.5, 8.5);
    REQUIRE(id_points.ids.size() == 1);
    REQUIRE(id_points.ids[0] == 2);
    REQUIRE(id_points.points.size() == 1);
    REQUIRE(id_points.points.nth(0).isApprox(Vector3d(3.0, 8.0, 13.0)));

    id_points = kd.range_query(2.0, 4.0, 6.0, 8.0);
    REQUIRE(id_points.ids.size() == 2);
    REQUIRE(id_points.ids[0] == 1);
    REQUIRE(id_points.ids[1] == 2);
    REQUIRE(id_points.points.size() == 2);
    REQUIRE(id_points.points.nth(0).isApprox(Vector3d(2.0, 7.0, 12.0)));
    REQUIRE(id_points.points.nth(1).isApprox(Vector3d(3.0, 8.0, 13.0)));
  };
}

