#include "cartesian_point_sources.hpp"

using Eigen::Vector3d;
using Eigen::Vector4d;

using thoracuda::CartesianPointSources;

CartesianPointSources::CartesianPointSources(std::string obscode) { this->obscode = obscode; }

CartesianPointSources::CartesianPointSources(std::string obscode, std::vector<double> x, std::vector<double> y,
                                             std::vector<double> z, std::vector<double> t) {
  this->obscode = obscode;
  this->x = x;
  this->y = y;
  this->z = z;
  this->t = t;
}

CartesianPointSources::CartesianPointSources(std::string obscode, int capacity) {
  this->obscode = obscode;
  this->x.reserve(capacity);
  this->y.reserve(capacity);
  this->z.reserve(capacity);
  this->t.reserve(capacity);
}

CartesianPointSources::~CartesianPointSources() {}

void CartesianPointSources::add(double x, double y, double z, double t) {
  this->x.push_back(x);
  this->y.push_back(y);
  this->z.push_back(z);
  this->t.push_back(t);
}

Vector3d CartesianPointSources::nth_pos(int n) {
  if (n >= this->size()) {
    throw std::out_of_range("CartesianPointSources::nth_pos: n is out of range");
  }
  if (n < 0) {
    throw std::out_of_range("CartesianPointSources::nth_pos: n is out of range");
  }
  return Vector3d(x[n], y[n], z[n]);
}

Vector4d CartesianPointSources::nth(int n) {
  if (n >= this->size()) {
    throw std::out_of_range("CartesianPointSources::nth: n is out of range");
  }
  if (n < 0) {
    throw std::out_of_range("CartesianPointSources::nth: n is out of range");
  }
  return Vector4d(x[n], y[n], z[n], t[n]);
}

int CartesianPointSources::size() { return x.size(); }

double CartesianPointSources::nth_t(int n) {
  if (n >= this->size()) {
    throw std::out_of_range("CartesianPointSources::nth_t: n is out of range");
  }
  if (n < 0) {
    throw std::out_of_range("CartesianPointSources::nth_t: n is out of range");
  }
  return t[n];
}
