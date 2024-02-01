#include <math.h>
#include <stdlib.h>

#include "rangearray_c.h"

int main() {
  int result;
  int n = 1000000;

  struct XYPairVector xyvec;
  result = xyvec_init(&xyvec, n);
  if (result != 0) {
    return result;
  }

  struct XYVectorBounds actual_bounds = {
      .xmin = INFINITY,
      .xmax = -INFINITY,
      .ymin = INFINITY,
      .ymax = -INFINITY,
  };
  for (int i = 0; i < n; i++) {
    xyvec.xy[i].x = rand() % 100000;
    xyvec.xy[i].y = rand() % 100000;
    if (xyvec.xy[i].x < actual_bounds.xmin) {
      actual_bounds.xmin = xyvec.xy[i].x;
    }
    if (xyvec.xy[i].x > actual_bounds.xmax) {
      actual_bounds.xmax = xyvec.xy[i].x;
    }
    if (xyvec.xy[i].y < actual_bounds.ymin) {
      actual_bounds.ymin = xyvec.xy[i].y;
    }
    if (xyvec.xy[i].y > actual_bounds.ymax) {
      actual_bounds.ymax = xyvec.xy[i].y;
    }
  }
  struct XYVectorBounds have_bounds;

  result = xyvec_bounds_parallel(&xyvec, &have_bounds);
  if (result != 0) {
    return result;
  }
  return 0;
}
