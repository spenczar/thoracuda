#pragma once
#include <Eigen/Dense>
#include <vector>

using Eigen::Vector3d;
using Eigen::Vector4d;

namespace thoracuda {
  class CartesianPointSources {
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    std::vector<double> t;  // MJD
   public:
    std::string obscode;
    CartesianPointSources(std::string obscode);
    CartesianPointSources(std::string obscode, std::vector<double> x, std::vector<double> y, std::vector<double> z,
                          std::vector<double> t);
    CartesianPointSources(std::string obscode, int capacity);
    ~CartesianPointSources();
    void add(double x, double y, double z, double t);
    Vector3d nth_pos(int n);
    Vector4d nth(int n);
    double nth_t(int n);
    int size();
  };
}  // namespace thoracuda
