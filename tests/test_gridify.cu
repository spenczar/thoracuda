#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include "cuda_macros.h"
#include "gridify.h"
#include "pair.h"


TEST_CASE("gridify_serial", "") {
  int N = 10000;
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
  // Should be 100 unique times, so there should be 64 * 64 * 100 = 409600 cells
  REQUIRE(grid.address_map_n == 409600);

  // Check that the grid is correct
  struct CellPosition cp;
  struct XYPair xy = {.x = 10, .y = 10};
  struct XYBounds bounds = {.xmin = 0.0f, .xmax = (float)N, .ymin=0.0f, .ymax=(float)N};
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
  REQUIRE(ip.end == 99);

  BENCHMARK("gridify_serial_10000") {
    free(grid.address_map);
    grid.address_map = NULL;
    return gridify_points_serial(xys, t, N, &grid);
  };
  
}

TEST_CASE("map_points_to_cells_kernel", "") {
  
  int N = 10000;
  struct XYPair xys[N];
  double t[N];
  struct CellPosition cps[N];
  for (int i = 0; i < N; i++) {
    xys[i].x = i;
    xys[i].y = i;
    t[i] = i / 100.0;
  }

  struct XYBounds bounds = {.xmin = 0.0f, .xmax = (float)N, .ymin=0.0f, .ymax=(float)N};
  double t_min = 0.0;
  
  cudaError_t err;
  struct XYPair *d_xys = NULL;
  double *d_t = NULL;
  struct CellPosition *d_cps = NULL;

  int block_size = 256;
  int n_blocks = (N + block_size - 1) / block_size;
  
  CUDA_OR_FAIL(cudaMalloc(&d_xys, N * sizeof(struct XYPair)));
  CUDA_OR_FAIL(cudaMemcpy(d_xys, xys, N * sizeof(struct XYPair), cudaMemcpyHostToDevice));
  
  CUDA_OR_FAIL(cudaMalloc(&d_t, N * sizeof(double)));
  CUDA_OR_FAIL(cudaMemcpy(d_t, t, N * sizeof(double), cudaMemcpyHostToDevice));

  CUDA_OR_FAIL(cudaMalloc(&d_cps, N * sizeof(struct CellPosition)));


  map_points_to_cells_kernel<<<n_blocks, block_size>>>(d_xys, d_t, N, bounds, t_min, d_cps);
  err = cudaGetLastError();
  REQUIRE(err == cudaSuccess);

  cudaDeviceSynchronize();
  
  CUDA_OR_FAIL(cudaMemcpy(cps, d_cps, N * sizeof(struct CellPosition), cudaMemcpyDeviceToHost));

  for (int i = 0; i < N; i++) {
    struct CellPosition cp = xy_to_cell(xys[i], bounds, t[i], t_min);
    REQUIRE(cp.x == cps[i].x);
    REQUIRE(cp.y == cps[i].y);
    REQUIRE(cp.t == cps[i].t);
  }
  

 fail:
  if (d_xys) {
    cudaFree(d_xys);
  }
  if (d_t) {
    cudaFree(d_t);
  }
  if (d_cps) {
    cudaFree(d_cps);
  }
  REQUIRE(err == cudaSuccess);
    
}