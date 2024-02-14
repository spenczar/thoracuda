#pragma once

#include <vector>
#include <map>

#include "kdtree.hpp"

using Eigen::Vector2d;

using thoracuda::GnomonicPointSources;

namespace thoracuda {
  namespace dbscan {

    enum PointType {
      UNCLASSIFIED,
      NOISE,
      CORE,
      BORDER
    };

    struct Cluster {
      std::vector<int> ids;
    };

    class DBSCAN {
      float eps;
      int min_size;
      GnomonicPointSources points;

     public:
      DBSCAN(float eps, int min_size, GnomonicPointSources points);
      std::vector<Cluster> fit();
    };

  }
}
