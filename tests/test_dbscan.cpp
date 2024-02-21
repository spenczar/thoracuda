#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <random>

#include <vector>
#include "gnomonic_point_sources.hpp"
#include "kdtree.hpp"
#include "dbscan.hpp"
#include "testutils.hpp"

using Eigen::Vector3d;
using thoracuda::GnomonicPointSources;
using thoracuda::dbscan::DBSCAN;
using thoracuda::dbscan::Cluster;

TEST_CASE("dbscan", "") {
  std::vector<double> x = {1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<double> y = {6.0, 6.0, 6.0, 6.0, 7.0, 8.0, 9.0, 10.0};
  std::vector<double> t = {11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0};

  GnomonicPointSources gps = GnomonicPointSources(x, y, t);

  float epsilon = 1.1;
  int min_points = 5;

  DBSCAN dbscan = DBSCAN(epsilon, min_points, gps);
  std::vector<Cluster> clusters = dbscan.fit();

  REQUIRE(clusters.size() == 1);
  REQUIRE(clusters[0].ids.size() == 6);

  // Benchmark with N=10000.
  //
  // True clusters at a few select locations, and then piles of noise.

  int N = 10000;
  epsilon = 0.1;
  float x_min = -10.0;
  float x_max = 10.0;
  float y_min = -10.0;
  float y_max = 10.0;
  std::random_device rd; 
  std::mt19937 gen(rd());

  std::uniform_real_distribution<> x_distr(x_min, x_max);
  std::uniform_real_distribution<> y_distr(y_min, y_max);
  std::uniform_int_distribution<> cluster_size_distr(min_points, 8*min_points);
  int N_cluster = 50;
  int total_cluster_points_generated = 0;
  for (int i = 0; i < N_cluster; i++) {
    float x_center = x_distr(gen);
    float y_center = y_distr(gen);
    int cluster_size = cluster_size_distr(gen);

    std::normal_distribution<> x_dist(x_center, epsilon/2.0);
    std::normal_distribution<> y_dist(y_center, epsilon/2.0);

    for (int j = 0; j < cluster_size; j++) {
      x.push_back(x_dist(gen));
      y.push_back(y_dist(gen));
      t.push_back(0.0);
    }
    total_cluster_points_generated += cluster_size;
  }
  // Bunch of uniform random points
  for (int i = 0; i < N - total_cluster_points_generated; i++) {
    x.push_back(x_distr(gen));
    y.push_back(y_distr(gen));
    t.push_back(0.0);
  }

  gps = GnomonicPointSources(x, y, t);

  BENCHMARK("dbscan") {
    DBSCAN dbscan = DBSCAN(epsilon, min_points, gps);
    std::vector<Cluster> clusters = dbscan.fit();
    return clusters;
  };
};

TEST_CASE("dbscan bench against THOR data", "") {
  
  GnomonicPointSources gps = read_point_data();
  REQUIRE(gps.size() == 11153);

  float eps = 0.002777777777777778;
  int min_size = 6;

  DBSCAN dbscan = DBSCAN(eps, min_size, gps);
  std::vector<Cluster> clusters = dbscan.fit();

  BENCHMARK("dbscan_on_thor") {
    DBSCAN dbscan = DBSCAN(eps, min_size, gps);
    std::vector<Cluster> clusters = dbscan.fit();
    return clusters;
  };
}
