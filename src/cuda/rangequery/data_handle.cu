#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

#include "cuda_macros.h"
#include "rangequery/row.cuh"
#include "rangequery/data_handle.cuh"

using thoracuda::rangequery::Row;

namespace thoracuda {
namespace rangequery {

  DataHandle::DataHandle(const std::vector<float> &x, const std::vector<float> &y, const std::vector<float> &t, const std::vector<int> &id) {
    n = x.size();
    if (x.size() != y.size() || x.size() != t.size() || x.size() != id.size()) {
      throw std::invalid_argument("All input vectors must be the same size");
    }

    std::vector<Row> rows;
    for (size_t i = 0; i < x.size(); i++) {
      rows.push_back({x[i], y[i], t[i], id[i]});
    }
    
    cudaError_t err;
    CUDA_OR_THROW(cudaMalloc(&this->rows, rows.size() * sizeof(Row)));
    CUDA_OR_THROW(cudaMemcpy(this->rows, rows.data(), rows.size() * sizeof(Row), cudaMemcpyHostToDevice));
  }
  
  DataHandle::~DataHandle() {
    if (rows != nullptr) {
      cudaFree(rows);
    }
  }

  DataHandle::DataHandle(DataHandle &&other) {
    rows = other.rows;
    n = other.n;

    other.rows = nullptr;
    other.n = 0;
  }

  DataHandle& DataHandle::operator=(DataHandle &&other) {
    if (this != &other) {
      rows = other.rows;
      n = other.n;

      other.rows = nullptr;
      other.n = 0;
    }
    return *this;
  }
    
  

}
}