#pragma once

#include <vector>

namespace thoracuda {
  // The result of a binmedian computation
  namespace stats {

    float binapprox(const std::vector<float>& values, float mean, float std_dev);
    float binmedian(const std::vector<float>& values, float mean, float std_dev);

    struct VectorStats {
      float mean;
      float std_dev;
      VectorStats(const std::vector<float>& values);
    };
  }
}
