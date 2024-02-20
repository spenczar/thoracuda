#pragma once

#include <vector>
#include <fstream>

#include "gnomonic_point_sources.hpp"

thoracuda::GnomonicPointSources read_point_data() {
  std::ifstream file("tests/benchdata/points.csv");
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file");
  }
  std::string line;
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> t;
  while (std::getline(file, line)) {
    std::string token;
    std::istringstream tokenStream(line);
    std::getline(tokenStream, token, ',');
    x.push_back(std::stod(token));
    std::getline(tokenStream, token, ',');
    y.push_back(std::stod(token));
    std::getline(tokenStream, token, ',');
    t.push_back(std::stod(token));
  }
  return thoracuda::GnomonicPointSources(x, y, t);
}
