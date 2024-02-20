#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include "stats.hpp"

using std::size_t;

using namespace thoracuda::stats;

#define BINAPPROX_BIN_COUNT 1024
#define BINMEDIAN_MIN_SIZE 32 // Below this size, use simple sort

VectorStats::VectorStats(const std::vector<float>& values) {
  if (values.size() == 0) {
    throw std::invalid_argument("Cannot create a Stats object with no values");
  }
  float s = 0;
  float m = values[0];
  float m_prev = m;

  for (size_t i = 0; i < values.size(); i++) {
    float p = values[i];
    if (i > 0) {
      m_prev = m;
      m = m_prev + (p - m_prev) / (i + 1);
      s = s + (p - m_prev) * (p - m);
    }
  }

  this->mean = m;
  float variance = s / (values.size() - 1);
  this->std_dev = sqrt(variance);
}
