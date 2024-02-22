#pragma  once

#include <vector>

namespace thoracuda {
namespace clustering {

struct Inputs {
  /// Inputs represents the (host-side) data that is provided for
  /// clustering.
  ///
  /// All vectors must have identical length.
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> t;
  std::vector<int> id;
};

}
}