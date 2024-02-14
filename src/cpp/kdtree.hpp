#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "gnomonic_point_sources.hpp"

using thoracuda::GnomonicPointSources;

namespace thoracuda {
  namespace kdtree {
    float median(std::vector<double> vals);
    enum SplitDimension { X, Y };
    
    struct IDPoint {
      int id;
      float x;
      float y;
      float t;
      IDPoint(int id, float x, float y, float t);
    };

    struct IDPoints {
      std::vector<int> ids;
      GnomonicPointSources points;

      // Merge two IDPoints objects
      IDPoints(std::vector<int> ids, GnomonicPointSources points);
      IDPoints(IDPoints p1, IDPoints p2);
      IDPoints();

      bool empty();
      IDPoint pop();
      void push(IDPoint p);
      void extend(IDPoints p);
      int size();
    };

    struct LeafNode {
      GnomonicPointSources points;
      std::vector<int> ids;
      LeafNode(GnomonicPointSources points, std::vector<int> ids);
      IDPoints range_query_x(float x_min, float x_max);
      IDPoints range_query_y(float y_min, float y_max);
    };

    struct KDNode {
      SplitDimension dim;
      float split;

      KDNode *left_kd;
      LeafNode *left_leaf;

      KDNode *right_kd;
      LeafNode *right_leaf;

      KDNode(SplitDimension dim, GnomonicPointSources points, std::vector<int> ids, int max_leaf_size = 16);
      ~KDNode();

      IDPoints range_query(float x_min, float x_max, float y_min, float y_max);
    };

    KDNode build_kdtree(GnomonicPointSources points, int max_leaf_size = 16);
  }  // namespace kdtree
}  // namespace thoracuda
