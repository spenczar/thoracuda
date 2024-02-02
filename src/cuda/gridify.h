#pragma once

#include "pair.h"
#include "pairminmax.h"

struct IndexPair {
  int start;
  int end;
};

struct Grid {
  struct IndexPair* address_map;
  int address_map_n;

  int n;
  struct XYPair *xys;
  double *ts;
};

struct CellPosition {
  short x;
  short y;
  short t;
};

struct CellPosition xy_to_cell(struct XYPair xy, struct XYBounds bounds, double t, double t_min);

struct SortableData {
  struct XYPair *xys;
  double *ts;

  struct XYBounds bounds;
  double t_min;
};

int xy_compare_txy(const void *a, const void *b, void *data);
  
int gridify_points_serial(struct XYPair* xys, double *t, int n, struct Grid *grid);

struct IndexPair grid_query(struct CellPosition cp, struct Grid *grid);
