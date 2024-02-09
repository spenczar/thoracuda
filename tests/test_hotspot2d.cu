#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>

#include "hotspot2d.h"
#include "pair.h"

using Catch::Approx;


void find_exposure_boundaries_serial(const float *ts, int n, int **boundaries, int *n_boundaries) {
  int n_boundaries_ = 0;
  for (int i = 1; i < n; i++) {
    if (ts[i] != ts[i - 1]) {
      n_boundaries_++;
    }
  }
  *n_boundaries = n_boundaries_;
  *boundaries = (int *)malloc(n_boundaries_ * sizeof(int));
  int j = 0;
  for (int i = 1; i < n; i++) {
    if (ts[i] != ts[i - 1]) {
      (*boundaries)[j] = i;
      j++;
    }
  }
}

TEST_CASE("Hotspot2D", "") {

  SKIP("not implemented yet");
  
  int n = 100;
  struct XYPair *pairs = (struct XYPair *)malloc(n * sizeof(struct XYPair));
  float *ts = (float *)malloc(n * sizeof(float));
  int *ids = (int *)malloc(n * sizeof(int));

  for (int i = 0; i < n; i++) {
    ids[i] = i;
  }

  // Simulated data has 10 "exposures" with random data scattered
  // between -10 and +10. Hidden amongst the random noise are two
  // "true objects" which at velocity (0.0, 0.0) and (+0.5, -0.5)
  // respectively. The true objects are at (-1.1, +1.1) and (-2.1,
  // +4.1) in the first timestamp. The times are 0.0, 0.1, 0.2, ...,
  float true_vel_x_1 = 0.0;
  float true_vel_y_1 = 0.0;
  float true_vel_x_2 = 0.5;
  float true_vel_y_2 = -0.5;
  for (int i = 0; i < n; i++) {
    float rand_x = (float)(rand() % 20 - 10);
    float rand_y = (float)(rand() % 20 - 10);
    ts[i] = (float)(i / 10);
    if (i % 10 == 0) {
      pairs[i].x = -1.1 + true_vel_x_1 * ts[i];
      pairs[i].y = 1.1 + true_vel_y_1 * ts[i];
    } else if (i % 10 == 1) {
      pairs[i].x = -2.1 + true_vel_x_2 * ts[i];
      pairs[i].y = 4.1 + true_vel_y_2 * ts[i];
    } else {
      pairs[i].x = rand_x;
      pairs[i].y = rand_y;
    }
  }

  struct Hotspot2DParams params = {
    .min_points = 5,
    .max_distance = 0.1,
    .max_rel_velocity = 1.0,
  };

  int n_clusters = 0;
  struct Cluster *clusters;

  cudaError_t err;

  err = hotspot2d_parallel(pairs, ts, ids, n, params, &clusters, &n_clusters);
  REQUIRE(err == cudaSuccess);

  REQUIRE(n_clusters == 2);
  REQUIRE(clusters[0].n == 10);
  REQUIRE(clusters[0].v_x == Approx(true_vel_x_1));
  REQUIRE(clusters[0].v_y == Approx(true_vel_y_1));

  for (int i = 0; i < clusters[0].n; i++) {
    REQUIRE(clusters[0].point_ids[i] == i * 10);
  }
  
  REQUIRE(clusters[1].n == 10);
  REQUIRE(clusters[1].v_x == Approx(true_vel_x_2));
  REQUIRE(clusters[1].v_y == Approx(true_vel_y_2));

  for (int i = 0; i < clusters[1].n; i++) {
    REQUIRE(clusters[1].point_ids[i] == i * 10 + 1);
  }

  free(pairs);
  free(ts);
  free(ids);
  free(clusters);
}

TEST_CASE("find exposure boundaries", "") {
  int n = 1'000'000;
  float ts[1'000'000];
  for (int i = 0; i < n; i++) {
    ts[i] = (float)(i / 10);
  }

  float *d_ts;
  REQUIRE(cudaMalloc(&d_ts, n * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_ts, ts, n * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);

  int *d_boundaries = NULL;
  int *d_n_boundaries;
  REQUIRE(cudaMalloc((void **)&d_n_boundaries, sizeof(int)) == cudaSuccess);

  cudaError_t err;
  err = find_exposure_boundaries(d_ts, n, &d_boundaries, d_n_boundaries);
  REQUIRE(err == cudaSuccess);

  int n_boundaries;
  REQUIRE(cudaMemcpy(&n_boundaries, d_n_boundaries, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
  int *boundaries = (int *)malloc(n_boundaries * sizeof(int));
  REQUIRE(cudaMemcpy(boundaries, d_boundaries, n_boundaries * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);

  BENCHMARK("find exposure boundaries") {
    return find_exposure_boundaries(d_ts, n, &d_boundaries, d_n_boundaries);
  };

  BENCHMARK("find exposure boundaries serial") {
    return find_exposure_boundaries_serial(ts, n, &boundaries, &n_boundaries);
  };

  cudaFree(d_ts);
  cudaFree(d_boundaries);
  cudaFree(d_n_boundaries);

  REQUIRE(n_boundaries == ((n/10) - 1));
  for (int i = 0; i < n_boundaries; i++) {
    REQUIRE(boundaries[i] == i * 10);
  }
  free(boundaries);
}