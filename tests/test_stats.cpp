#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include <algorithm>
#include <random>
#include <vector>
#include "stats.hpp"


using Catch::Matchers::WithinRel;

using thoracuda::stats::VectorStats;
using thoracuda::stats::binapprox;
using thoracuda::stats::binmedian;



TEST_CASE("VectorStats", "") {
  std::vector<float> values = {1, 2, 3, 4, 5};
  VectorStats stats(values);
  REQUIRE(stats.mean == 3);
  REQUIRE_THAT(stats.std_dev, WithinRel(1.58114, 1e-5));

  values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  stats = VectorStats(values);
  REQUIRE(stats.mean == 5.5);
  REQUIRE_THAT(stats.std_dev, WithinRel(3.02765, 1e-5));

  values = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  stats = VectorStats(values);
  REQUIRE(stats.mean == 0);
  REQUIRE_THAT(stats.std_dev, WithinRel(0, 1e-5));

  values = {};
  REQUIRE_THROWS(VectorStats(values));

  std::vector<float> values2(100000, 0);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> d(0, 1);
  for (int i = 0; i < 100000; i++) {
    values2[i] = d(gen);
  }
  BENCHMARK("VectorStats of 100,000 values") {
    return VectorStats(values2);
  };
}

TEST_CASE("binmedian", "") {
  std::vector<float> values = {1, 2, 3, 4, 5};
  float mean = 3;
  float std_dev = 1.58114;
  REQUIRE(binmedian(values, mean, std_dev) == 3);

  int N = 100001;
  values = std::vector<float>(N, 0);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> d(0, 1);
  for (int i = 0; i < N; i++) {
    values[i] = d(gen);
  }

  VectorStats stats(values);
  float have = binmedian(values, stats.mean, stats.std_dev);

  std::sort(values.begin(), values.end());
  float want = values[N / 2];
  REQUIRE_THAT(have, WithinRel(want, 1e-8f));

  BENCHMARK("binmedian of 100,000 values") {
    VectorStats stats(values);
    return binmedian(values, stats.mean, stats.std_dev);
  };
  BENCHMARK_ADVANCED("sort median of 100,000 values")(Catch::Benchmark::Chronometer meter) {
    std::vector<float> values_copy = values;
    meter.measure([&values_copy, N] {
      std::sort(values_copy.begin(), values_copy.end());
      return values_copy[N / 2];
    });
  };
}
