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

    OffsetsTable::OffsetsTable(const DataHandle &data, const RangeQueryParams &params) {
      cudaError_t err;

      // First, count the number of results for each point
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

      // Prefix sum to get the offsets
      int *offsets;
      CUDA_OR_THROW(cudaMalloc(&offsets, data.n * sizeof(int)));
      void *temp_storage_d = NULL;
      size_t temp_storage_bytes = 0;
      CUDA_OR_THROW(cub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes, counts, offsets, data.n));
      CUDA_OR_THROW(cudaMalloc(&temp_storage_d, temp_storage_bytes));
      CUDA_OR_THROW(cub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes, counts, offsets, data.n));
      cudaFree(temp_storage_d);
      CUDA_OR_THROW(cudaGetLastError());


      this->offsets = offsets;
      this->n_offsets = data.n;
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

    int OffsetsTable::n_total_results() {
      cudaError_t err;
      int last_offset;
      CUDA_OR_THROW(cudaMemcpy(&last_offset, this->offsets + this->n_offsets - 1, sizeof(int), cudaMemcpyDeviceToHost));
      return last_offset;
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

  }
}