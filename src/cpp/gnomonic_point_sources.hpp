#pragma once
#include <Eigen/Dense>
#include <vector>

using Eigen::Vector3d;

namespace thoracuda {
  class GnomonicPointSources {
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> t;  // MJD
   public:
    GnomonicPointSources();
    GnomonicPointSources(std::vector<double> x, std::vector<double> y, std::vector<double> t);
    GnomonicPointSources(int capacity);
    ~GnomonicPointSources();

    void add(double x, double y, double t);
    Vector3d nth(int n);
    int size();
  };
}  // namespace thoracuda
