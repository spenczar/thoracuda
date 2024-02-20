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

static float bin_spacing(float std_dev) {
  return (BINAPPROX_BIN_COUNT) / (2 * std_dev);
}

float thoracuda::stats::binmedian(const std::vector<float>& values, float mean, float std_dev) {
  // Compute the median using the binapprox algorithm
  // (Tibshirani 2008):
  // https://www.stat.cmu.edu/~ryantibs/papers/median/
  std::vector<float> vals = values;

  int target_rank = (vals.size() + 1) / 2;
  while (vals.size() > BINMEDIAN_MIN_SIZE) {
    int under_bins = 0;
    int bins[BINAPPROX_BIN_COUNT];
    for (size_t i = 0; i < BINAPPROX_BIN_COUNT; i++) {
      bins[i] = 0;
    }
    float bin_min = mean - std_dev;
    float bin_max = mean + std_dev;

    float bin_width = bin_spacing(std_dev);

    // Map the points to bins, building a little histogram of the
    // points around mean +/- std_dev
    for (size_t i = 0; i < vals.size(); i++) {
      float p = vals[i];
      if (p < bin_min) {
	under_bins++;
      } else if (p <= bin_max) {
	int bin = (int) ((p - bin_min) * bin_width);
	bins[bin]++;
      }
    }

    // Find the bin where we cross the median count threshold
    int have_total = under_bins;
    int target_bin = -1;
    for (size_t i = 0; i < BINAPPROX_BIN_COUNT; i++) {
      have_total += bins[i];
      if (have_total >= target_rank) {
	target_bin = i;
	break;
      }
    }
    if (target_bin == -1) {
      // This should never happen, but just in case
      throw std::runtime_error("binmedian: no target bin found");
    }
    // Offset the target rank - we don't want to find the median of
    // the bin, but rather some k-th value of it.
    target_rank = target_rank - (have_total - bins[target_bin]);
    

    // Find the values that fall into the target bin
    float bin_start = bin_min + target_bin / bin_width;
    float bin_end = bin_min + (target_bin + 1) / bin_width;

    std::vector<float> next_vals;
    next_vals.reserve(bins[target_bin]);
    for (size_t i = 0; i < vals.size(); i++) {
      float p = vals[i];
      if (p >= bin_start && p < bin_end) {
	next_vals.push_back(p);
      }
    }

    if (next_vals.size() == 0) {
      // This should never happen, but just in case
      throw std::runtime_error("binmedian: no values in target bin");
    }
    vals = next_vals;
    VectorStats stats(vals);
    mean = stats.mean;
    std_dev = stats.std_dev;
    if (std_dev == 0) {
      // All values are the same.
      return vals[0];
    }
    
  }

  // Sort the remaining values and find the median
  std::sort(vals.begin(), vals.end());
  return vals[target_rank - 1];
}

float thoracuda::stats::binapprox(const std::vector<float>& values, float mean, float std_dev) {
  // Compute the median using the binapprox algorithm
  // (Tibshirani 2008):
  // https://www.stat.cmu.edu/~ryantibs/papers/median/
  int under_bins = 0;
  int bins[BINAPPROX_BIN_COUNT];
  for (size_t i = 0; i < BINAPPROX_BIN_COUNT; i++) {
    bins[i] = 0;
  }
  float bin_min = mean - std_dev;
  float bin_max = mean + std_dev;

  float bin_width = bin_spacing(std_dev);

  // Map the points to bins, building a little histogram of the
  // points around mean +/- std_dev
  for (size_t i = 0; i < values.size(); i++) {
    float p = values[i];
    if (p < bin_min) {
      under_bins++;
    } else if (p < bin_max) {
      int bin = (int) ((p - bin_min) * bin_width);
      bins[bin]++;
    }
  }

  // Find the bin where we cross the median count threshold
  int target_total = (values.size() + 1) / 2;
  int have_total = under_bins;
  int target_bin = 0;
  for (size_t i = 0; i < BINAPPROX_BIN_COUNT; i++) {
    have_total += bins[i];
    if (have_total >= target_total) {
      target_bin = i;
      break;
    }
  }
  
  // Interpolate the bin to find the median
  int prev_total = have_total - bins[target_bin];
  int target_excess = target_total - prev_total;
  float fractional_bin = (float) target_excess / bins[target_bin];

  return bin_min + (target_bin + fractional_bin) / bin_width;
}
