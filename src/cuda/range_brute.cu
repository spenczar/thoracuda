#include <cub/device/device_scan.cuh>

#include "range_brute.h"

#include "cuda_macros.h"

using namespace thoracuda::rangequery;

namespace thoracuda {
  namespace rangequery {

    DataHandle::DataHandle(const Inputs &inputs) {
      cudaError_t err;
      CUDA_OR_THROW(cudaMalloc(&x, inputs.x.size() * sizeof(float)));
      CUDA_OR_THROW(cudaMemcpy(x, inputs.x.data(), inputs.x.size() * sizeof(float), cudaMemcpyHostToDevice));

      CUDA_OR_THROW(cudaMalloc(&y, inputs.y.size() * sizeof(float)));
      CUDA_OR_THROW(cudaMemcpy(y, inputs.y.data(), inputs.y.size() * sizeof(float), cudaMemcpyHostToDevice));

      CUDA_OR_THROW(cudaMalloc(&t, inputs.t.size() * sizeof(float)));
      CUDA_OR_THROW(cudaMemcpy(t, inputs.t.data(), inputs.t.size() * sizeof(float), cudaMemcpyHostToDevice));

      CUDA_OR_THROW(cudaMalloc(&id, inputs.id.size() * sizeof(int)));
      CUDA_OR_THROW(cudaMemcpy(id, inputs.id.data(), inputs.id.size() * sizeof(int), cudaMemcpyHostToDevice));

      n = inputs.x.size();
    }

    DataHandle::~DataHandle() {
      if (id != nullptr) {
	cudaFree(id);
      }
      if (t != nullptr) {
	cudaFree(t);
      }
      if (y != nullptr) {
	cudaFree(y);
      }
      if (x != nullptr) {
	cudaFree(x);
      }
    }

    DataHandle::DataHandle(DataHandle &&other) {
      x = other.x;
      y = other.y;
      t = other.t;
      id = other.id;
      n = other.n;

      other.x = nullptr;
      other.y = nullptr;
      other.t = nullptr;
      other.id = nullptr;
      other.n = 0;
    }

    DataHandle& DataHandle::operator=(DataHandle &&other) {
      if (this != &other) {
	x = other.x;
	y = other.y;
	t = other.t;
	id = other.id;
	n = other.n;

	other.x = nullptr;
	other.y = nullptr;
	other.t = nullptr;
	other.id = nullptr;
	other.n = 0;
      }
      return *this;
    }
    
    CountsTable::CountsTable() {
      counts = nullptr;
      n_counts = 0;
    }

    CountsTable::CountsTable(const DataHandle &data, const RangeQueryParams &params) {
      cudaError_t err;

      // Count the number of results for each point
      int *counts;
      CUDA_OR_THROW(cudaMalloc(&counts, data.n * sizeof(int)));

      int threads_per_block = 256;
      int blocks = (data.n + threads_per_block - 1) / threads_per_block;

      if (params.metric == RangeQueryMetric::EUCLIDEAN) {
	count_within_euclidean_distance<<<blocks, threads_per_block>>>(data.x, data.y, data.t, data.n, params.distance * params.distance, counts);
      } else if (params.metric == RangeQueryMetric::MANHATTAN) {
	count_within_manhattan_distance<<<blocks, threads_per_block>>>(data.x, data.y, data.t, data.n, params.distance, counts);
      } else {
	throw std::runtime_error("Unsupported metric");
      }

      this->counts = counts;
      this->n_counts = data.n;
    }

    CountsTable::~CountsTable() {
      if (counts != nullptr) {
	cudaFree(counts);
      }
    }

    CountsTable::CountsTable(CountsTable &&other) {
      counts = other.counts;
      n_counts = other.n_counts;

      other.counts = nullptr;
      other.n_counts = 0;
    }

    CountsTable& CountsTable::operator=(CountsTable &&other) {
      if (this != &other) {
	counts = other.counts;
	n_counts = other.n_counts;

	other.counts = nullptr;
	other.n_counts = 0;
      }
      return *this;
    }

      OffsetsTable::OffsetsTable(const CountsTable &counts) {
	this->initialize_from_counts(counts);
      }

    void OffsetsTable::initialize_from_counts(const CountsTable &counts) {
      cudaError_t err;
      // Prefix sum to get the offsets
      int *offsets;
      CUDA_OR_THROW(cudaMalloc(&offsets, counts.n_counts * sizeof(int)));
      void *temp_storage_d = NULL;
      size_t temp_storage_bytes = 0;
      CUDA_OR_THROW(cub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes, counts.counts, offsets, counts.n_counts));
      CUDA_OR_THROW(cudaMalloc(&temp_storage_d, temp_storage_bytes));
      CUDA_OR_THROW(cub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes, counts.counts, offsets, counts.n_counts));
      cudaFree(temp_storage_d);
      CUDA_OR_THROW(cudaGetLastError());

      this->offsets = offsets;
      this->n_offsets = counts.n_counts;
    }

    OffsetsTable::OffsetsTable(const DataHandle &data, const RangeQueryParams &params) {
      CountsTable counts(data, params);
      this->initialize_from_counts(counts);
    }

    OffsetsTable::OffsetsTable() {
      offsets = nullptr;
      n_offsets = 0;
    }
    
    OffsetsTable::~OffsetsTable() {
      if (offsets != nullptr) {
	cudaFree(offsets);
      }
    }

    OffsetsTable::OffsetsTable(OffsetsTable &&other) {
      offsets = other.offsets;
      n_offsets = other.n_offsets;

      other.offsets = nullptr;
      other.n_offsets = 0;
    }

    OffsetsTable& OffsetsTable::operator=(OffsetsTable &&other) {
      if (this != &other) {
	offsets = other.offsets;
	n_offsets = other.n_offsets;

	other.offsets = nullptr;
	other.n_offsets = 0;
      }
      return *this;
    }

    int OffsetsTable::n_total_results() const {
      cudaError_t err;
      int last_offset;
      CUDA_OR_THROW(cudaMemcpy(&last_offset, this->offsets + this->n_offsets - 1, sizeof(int), cudaMemcpyDeviceToHost));
      return last_offset;
    }

    NeighborsTable::NeighborsTable() {
      neighbors = nullptr;
      n_neighbors = 0;
    }

    NeighborsTable::NeighborsTable(const DataHandle &data, const RangeQueryParams &params, const OffsetsTable &offsets) {
      cudaError_t err;

      // Count the number of results for each point
      int *neighbors;
      this->n_neighbors = offsets.n_total_results();
      CUDA_OR_THROW(cudaMalloc(&neighbors, this->n_neighbors * sizeof(int)));

      int threads_per_block = 256;
      int blocks = (data.n + threads_per_block - 1) / threads_per_block;

      if (params.metric == RangeQueryMetric::EUCLIDEAN) {
	range_query_euclidean<<<blocks, threads_per_block>>>(data.x, data.y, data.t, data.id, data.n, params.distance * params.distance, offsets.offsets, neighbors);
      } else if (params.metric == RangeQueryMetric::MANHATTAN) {
	range_query_manhattan<<<blocks, threads_per_block>>>(data.x, data.y, data.t, data.id, data.n, params.distance, offsets.offsets, neighbors);
      } else {
	throw std::runtime_error("Unsupported metric");
      }

      this->neighbors = neighbors;
    }

    NeighborsTable::~NeighborsTable() {
      if (neighbors != nullptr) {
	cudaFree(neighbors);
      }
    }

    NeighborsTable::NeighborsTable(NeighborsTable &&other) {
      neighbors = other.neighbors;
      n_neighbors = other.n_neighbors;

      other.neighbors = nullptr;
      other.n_neighbors = 0;
    }

    NeighborsTable& NeighborsTable::operator=(NeighborsTable &&other) {
      if (this != &other) {
	neighbors = other.neighbors;
	n_neighbors = other.n_neighbors;

	other.neighbors = nullptr;
	other.n_neighbors = 0;
      }
      return *this;
    }

    Results::Results(int *results, int n_results, OffsetsTable offsets) {
      this->results = results;
      this->n_results = n_results;
      this->offsets = std::move(offsets);
    }

    Results::~Results() {
      if (results != nullptr) {
	cudaFree(results);
      }
      if (offsets.offsets != nullptr) {
	
      }
    }

    Results::Results(Results &&other) {
      results = other.results;
      n_results = other.n_results;
      offsets = std::move(other.offsets);

      other.results = nullptr;
      other.n_results = 0;
    }

    Results& Results::operator=(Results &&other) {
      if (this != &other) {
	results = other.results;
	n_results = other.n_results;
	offsets = std::move(other.offsets);

	other.results = nullptr;
	other.n_results = 0;
      }
      return *this;
    }

    Results range_query(const Inputs &inputs, const RangeQueryParams &params) {
      cudaError_t err;

      DataHandle data(inputs);
      OffsetsTable offsets(data, params);

      int *result;
      int n_results = offsets.n_total_results();
      CUDA_OR_THROW(cudaMalloc(&result, n_results * sizeof(int)));

      int threads_per_block = 256;
      int blocks = (data.n + threads_per_block - 1) / threads_per_block;

      if (params.metric == RangeQueryMetric::EUCLIDEAN) {
	range_query_euclidean<<<blocks, threads_per_block>>>(data.x, data.y, data.t, data.id, data.n, params.distance * params.distance, offsets.offsets, result);
      } else if (params.metric == RangeQueryMetric::MANHATTAN) {
	range_query_manhattan<<<blocks, threads_per_block>>>(data.x, data.y, data.t, data.id, data.n, params.distance, offsets.offsets, result);
      } else {
	throw std::runtime_error("Unsupported metric");
      }

      return {result, n_results, std::move(offsets)};
    }

    std::vector<std::vector<int>> Results::get_results() {
      cudaError_t err;

      std::vector<std::vector<int>> results;
      int n_offsets = this->offsets.n_offsets;
      results.resize(n_offsets);

      int *host_results = new int[n_results];
      CUDA_OR_THROW(cudaMemcpy(host_results, this->results, n_results * sizeof(int), cudaMemcpyDeviceToHost));

      int *host_offsets = new int[n_offsets];

      CUDA_OR_THROW(cudaMemcpy(host_offsets, this->offsets.offsets, n_offsets * sizeof(int), cudaMemcpyDeviceToHost));

      for (int i = 0; i < n_offsets; i++) {
	// Subtle: this indexing scheme is tightly linked to use of
	// exclusive vs inclusive scan
	int start = i == 0 ? 0 : host_offsets[i - 1];
	int end = host_offsets[i];
	results[i].resize(end - start);
	for (int j = start; j < end; j++) {
	  results[i][j - start] = host_results[j];
	}
      }
      delete[] host_results;
      return results;
    }

    __global__ void count_within_euclidean_distance(float *x, float *y, float *t, int n, float sq_distance, int *result) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      for (; i < n; i += blockDim.x * gridDim.x) {
	int count = 0;
	float xi = x[i];
	float yi = y[i];
	float ti = t[i];
	for (int j = 0; j < n; j++) {
	  if (t[j] == ti) {
	    continue;
	  }
	  float dx = xi - x[j];
	  float dy = yi - y[j];
	  float sq_dist = dx * dx + dy * dy;
	  if (sq_dist <= sq_distance) {
	    count++;
	  }
	}
	result[i] = count;
      }
    }

    __global__ void count_within_manhattan_distance(float *x, float *y, float *t, int n, float distance, int *result) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      for (; i < n; i += blockDim.x * gridDim.x) {
	int count = 0;
	float xi = x[i];
	float yi = y[i];
	float ti = t[i];

	float xmin = xi - distance / 2.0;
	float xmax = xi + distance / 2.0;
	float ymin = yi - distance / 2.0;
	float ymax = yi + distance / 2.0;
	for (int j = 0; j < n; j++) {
	  if (t[j] == ti) {
	    continue;
	  }
	  float xj = x[j];
	  float yj = y[j];
	  if (xj >= xmin && xj <= xmax && yj >= ymin && yj <= ymax) {
	    count++;
	  }
	}
	result[i] = count;
      }
    }
    
    __global__ void range_query_euclidean(float *x, float *y, float *t, int *id, int n, float sq_distance, int *offsets, int *result) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      for (; i < n; i += blockDim.x * gridDim.x) {
	int count = 0;
	float xi = x[i];
	float yi = y[i];
	float ti = t[i];
	// Subtle: this indexing scheme is tightly linked to use of
	// exclusive vs inclusive scan
	int start = i == 0 ? 0 : offsets[i - 1];
	for (int j = 0; j < n; j++) {
	  if (t[j] == ti) {
	    continue;
	  }
	  float dx = xi - x[j];
	  float dy = yi - y[j];
	  float sq_dist = dx * dx + dy * dy;
	  if (sq_dist <= sq_distance) {
	    result[start + count] = id[j];
	    count++;
	  }
	}
      }
    }

    __global__ void range_query_manhattan(float *x, float *y, float *t, int *id, int n, float distance, int *offsets, int *result) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      for (; i < n; i += blockDim.x * gridDim.x) {
	int count = 0;
	float xi = x[i];
	float yi = y[i];
	float ti = t[i];
	int start = offsets[i];
	float xmin = xi - distance / 2.0;
	float xmax = xi + distance / 2.0;
	float ymin = yi - distance / 2.0;
	float ymax = yi + distance / 2.0;
	for (int j = 0; j < n; j++) {
	  if (t[j] == ti) {
	    continue;
	  }
	  float xj = x[j];
	  float yj = y[j];
	  if (xj >= xmin && xj <= xmax && yj >= ymin && yj <= ymax) {
	    result[start + count] = id[j];
	    count++;
	  }
	}
      }
    }

    std::vector<int> DBSCAN(float epsilon, int min_points, const Inputs &inputs) {
      /*
	Overall strategy is:
	1. Count points withiin epsilon of each point.
	
	2. For each point, if it has at least min_points neighbors,
	mark it as a core point. Do this by marking its cluster ID as
	its own ID.
	
	3. For each point with 0 < n < min_points neighbors,
	"associate" the point to a neighboring core point. If no
	neighbors, it is noise, and gets cluster ID -1.
	
	4. For each point, "pointer chase" through its neighbors to
	  find the smallest id, which is the cluster ID that it is
	  part of.  This is a recursive process.
	
       */
      cudaError_t err;

      // 1. Count points within epsilon.
      DataHandle data(inputs);
      RangeQueryParams params = {RangeQueryMetric::EUCLIDEAN, epsilon};

      CountsTable counts(data, params);

      // 2. Identify core points
      int *cluster_ids;
      CUDA_OR_THROW(cudaMalloc(&cluster_ids, data.n * sizeof(int)));
      int threads_per_block = 256;
      int blocks = (data.n + threads_per_block - 1) / threads_per_block;
      identify_core_points<<<blocks, threads_per_block>>>(counts.counts, counts.n_counts, min_points, cluster_ids);

      // 3. Merge core points into clusters
      OffsetsTable offsets(counts);
      NeighborsTable neighbors(data, params, offsets);
      collapse_clusters<<<blocks, threads_per_block>>>(cluster_ids, data.n, offsets.offsets, counts.counts, neighbors.neighbors);

      // 4. Associate border points to clusters
      associate_points_to_core_points<<<blocks, threads_per_block>>>(counts.counts, counts.n_counts, offsets.offsets, neighbors.neighbors, cluster_ids, min_points);

      std::vector<int> host_cluster_ids(data.n);
      CUDA_OR_THROW(cudaMemcpy(host_cluster_ids.data(), cluster_ids, data.n * sizeof(int), cudaMemcpyDeviceToHost));

      cudaFree(cluster_ids);

      return host_cluster_ids;
    }

    __global__ void identify_core_points(int *counts, int n, int min_points, int *cluster_ids) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      for (; i < n; i += blockDim.x * gridDim.x) {
	if (counts[i] >= (min_points-1)) {
	  cluster_ids[i] = i;
	} else {
	  cluster_ids[i] = -1;
	}
      }
    }
    
    __global__ void associate_points_to_core_points(int *counts, int n, int *offsets, int *neighbors, int *cluster_ids, int min_points) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      for (; i < n; i += blockDim.x * gridDim.x) {
	if (counts[i] > 0 && counts[i] < (min_points-1)) {
	  int start = i == 0 ? 0 : offsets[i - 1];
	  int end = offsets[i];
	  int cluster_id = -1;
	  for (int j = start; j < end; j++) {
	    int neighbor_id = neighbors[j];
	    if (cluster_ids[neighbor_id] != -1) {
	      cluster_id = cluster_ids[neighbor_id];
	      break;
	    }
	  }
	  cluster_ids[i] = cluster_id;
	}
      }
    }

    __global__ void collapse_clusters(int *cluster_ids, int n, int *offsets, int *counts, int *neighbors) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      for (; i < n; i += blockDim.x * gridDim.x) {
	int cluster_id = cluster_ids[i];
	if (cluster_id == -1) {
	  continue;
	}
	int ids_to_update[16];
	int n_ids_to_update = 0;

	// Recursively find the root of the cluster
	while (true) {
	  int merge_into = min_neighbor_cluster_id(i, cluster_ids, offsets, counts, neighbors);
	  if (merge_into == cluster_id) {
	    // Found the root
	    break;
	  }
	  if (n_ids_to_update < 16) {
	    ids_to_update[n_ids_to_update] = i;
	    n_ids_to_update++;
	  }
	  cluster_id = merge_into;
	}

	if (n_ids_to_update > 0) {
	  for (int j = 0; j < n_ids_to_update; j++) {
	    cluster_ids[ids_to_update[j]] = cluster_id;
	  }
	}
      }
    }

    __device__ int min_neighbor_cluster_id(int i, int *cluster_ids, int *offsets, int *counts, int *neighbors) {
      int cluster_id = cluster_ids[i];
      if (cluster_id == -1) {
	return -1;
      }
      // Subtle: this indexing scheme is tightly linked to use of
      // exclusive vs inclusive scan
      int start = i == 0 ? 0 : offsets[i - 1];
      int end = offsets[i];
      for (int j = start; j < end; j++) {
	int neighbor_id = neighbors[j];
	int neighbor_cluster_id = cluster_ids[neighbor_id];
	if (neighbor_cluster_id != -1 && neighbor_cluster_id < cluster_id) {
	  cluster_id = neighbor_cluster_id;
	}
      }
      return cluster_id;
    }

  }
}