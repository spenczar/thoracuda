#include "gnomonic_point_sources.hpp"

using Eigen::Vector3d;
using thoracuda::GnomonicPointSources;

GnomonicPointSources::GnomonicPointSources() {}

GnomonicPointSources::GnomonicPointSources(std::vector<double> x, std::vector<double> y, std::vector<double> t) {
  this->x = x;
  this->y = y;
  this->t = t;
}

GnomonicPointSources::GnomonicPointSources(int capacity) {
  this->x.reserve(capacity);
  this->y.reserve(capacity);
  this->t.reserve(capacity);
}

GnomonicPointSources::~GnomonicPointSources() {}

void GnomonicPointSources::add(double x, double y, double t) {
  this->x.push_back(x);
  this->y.push_back(y);
  this->t.push_back(t);
}

Vector3d GnomonicPointSources::nth(int n) {
  if (n >= this->size()) {
    throw std::out_of_range("GnomonicPointSources::nth: n is out of range");
  }
  if (n < 0) {
    throw std::out_of_range("GnomonicPointSources::nth: n is out of range");
  }
  return Vector3d(this->x[n], this->y[n], this->t[n]);
}

int GnomonicPointSources::size() { return this->x.size(); }
