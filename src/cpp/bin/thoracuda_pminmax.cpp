#include <math.h>
#include <stdlib.h>

#include "rangearray_c.h"

int main() {
  int result;
  int n = 1000000;

  struct XYPair *xys = (struct XYPair *)(malloc(n * sizeof(struct XYPair)));
  if (xys == NULL) {
    return 1;
  }
  struct XYBounds actual_bounds = {
      .xmin = INFINITY,
      .xmax = -INFINITY,
      .ymin = INFINITY,
      .ymax = -INFINITY,
  };
  for (int i = 0; i < n; i++) {
    xys[i].x = rand() % 100000;
    xys[i].y = rand() % 100000;
    if (xys[i].x < actual_bounds.xmin) {
      actual_bounds.xmin = xys[i].x;
    }
    if (xys[i].x > actual_bounds.xmax) {
      actual_bounds.xmax = xys[i].x;
    }
    if (xys[i].y < actual_bounds.ymin) {
      actual_bounds.ymin = xys[i].y;
    }
    if (xys[i].y > actual_bounds.ymax) {
      actual_bounds.ymax = xys[i].y;
    }
  }
  struct XYBounds have_bounds;

  result = xy_bounds_parallel(xys, n, &have_bounds);
  if (result != 0) {
    return result;
  }
  return 0;
}
