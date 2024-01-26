#include <Eigen/Dense>
#include <iostream>

#include "projections.hpp"

using Eigen::MatrixXd;

namespace tcp = thoracuda::projections;

int main(int argc, char** argv) {
  MatrixXd m(2, 2);
  m(0, 0) = 3;
  m(1, 0) = 2.5;
  m(0, 1) = -1;
  m(1, 1) = m(1, 0) + m(0, 1);
  std::cout << m << std::endl;

  Eigen::Vector3d center_pos = {1.0, 0.3, 0.1};
  Eigen::Vector3d center_vel = {0.0, 1.0, 0.2};

  Eigen::Matrix3d r1 = tcp::r1_matrix(center_pos, center_vel);
  std::cout << r1 << std::endl;
  return 0;
}
