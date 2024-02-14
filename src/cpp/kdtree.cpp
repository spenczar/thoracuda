#include "kdtree.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include "gnomonic_point_sources.hpp"

using thoracuda::GnomonicPointSources;
using namespace thoracuda::kdtree;

LeafNode::LeafNode(GnomonicPointSources points, std::vector<int> ids) {
  this->points = points;
  this->ids = ids;
}

IDPoints LeafNode::range_query_x(float x_min, float x_max) {
  GnomonicPointSources points(this->points.size());
  std::vector<int> ids;
  for (int i = 0; i < this->points.size(); i++) {
    if (x_min <= this->points.x[i] && this->points.x[i] <= x_max) {
      points.add(this->points.x[i], this->points.y[i], this->points.t[i]);
      ids.push_back(this->ids[i]);
    }
  }
  return IDPoints(ids, points);
}

IDPoints LeafNode::range_query_y(float y_min, float y_max) {
  GnomonicPointSources points(this->points.size());
  std::vector<int> ids;
  for (int i = 0; i < this->points.size(); i++) {
    if (y_min <= this->points.y[i] && this->points.y[i] <= y_max) {
      points.add(this->points.x[i], this->points.y[i], this->points.t[i]);
      ids.push_back(this->ids[i]);
    }
  }
  return IDPoints(ids, points);
}

IDPoints::IDPoints() {
  this->ids = std::vector<int>();
  this->points = GnomonicPointSources();
}

IDPoints::IDPoints(std::vector<int> ids, GnomonicPointSources points) {
  this->ids = ids;
  this->points = points;
}

IDPoints::IDPoints(IDPoints p1, IDPoints p2) {
  this->ids = p1.ids;
  this->ids.insert(this->ids.end(), p2.ids.begin(), p2.ids.end());
  this->points = p1.points;
  this->points.x.insert(this->points.x.end(), p2.points.x.begin(), p2.points.x.end());
  this->points.y.insert(this->points.y.end(), p2.points.y.begin(), p2.points.y.end());
  this->points.t.insert(this->points.t.end(), p2.points.t.begin(), p2.points.t.end());
}

bool IDPoints::empty() {
  return this->ids.empty();
}

IDPoint IDPoints::pop() {
  int id = this->ids.back();
  this->ids.pop_back();
  float x = this->points.x.back();
  this->points.x.pop_back();
  float y = this->points.y.back();
  this->points.y.pop_back();
  float t = this->points.t.back();
  this->points.t.pop_back();
  return IDPoint(id, x, y, t);
}

void IDPoints::extend(IDPoints p) {
  this->ids.insert(this->ids.end(), p.ids.begin(), p.ids.end());
  this->points.x.insert(this->points.x.end(), p.points.x.begin(), p.points.x.end());
  this->points.y.insert(this->points.y.end(), p.points.y.begin(), p.points.y.end());
  this->points.t.insert(this->points.t.end(), p.points.t.begin(), p.points.t.end());
}

void IDPoints::push(IDPoint p) {
  this->ids.push_back(p.id);
  this->points.add(p.x, p.y, p.t);
}

int IDPoints::size() {
  return this->ids.size();
}

IDPoint::IDPoint(int id, float x, float y, float t) {
  this->id = id;
  this->x = x;
  this->y = y;
  this->t = t;
}

KDNode::KDNode(SplitDimension dim, GnomonicPointSources points, std::vector<int> ids, int max_leaf_size) {
  this->dim = dim;
  this->left_kd = nullptr;
  this->left_leaf = nullptr;
  this->right_kd = nullptr;
  this->right_leaf = nullptr;

  GnomonicPointSources left_points;
  std::vector<int> left_ids;
  GnomonicPointSources right_points;
  std::vector<int> right_ids;

  SplitDimension next_dim;

  if (dim == SplitDimension::X) {
    next_dim = SplitDimension::Y;
    this->split = median(points.x);
    for (int i = 0; i < points.size(); i++) {
      if (points.x[i] < this->split) {
        left_points.add(points.x[i], points.y[i], points.t[i]);
        left_ids.push_back(ids[i]);
      } else {
        right_points.add(points.x[i], points.y[i], points.t[i]);
        right_ids.push_back(ids[i]);
      }
    }
  } else {
    next_dim = SplitDimension::X;
    this->split = median(points.y);
    for (int i = 0; i < points.size(); i++) {
      if (points.y[i] < this->split) {
        left_points.add(points.x[i], points.y[i], points.t[i]);
        left_ids.push_back(ids[i]);
      } else {
        right_points.add(points.x[i], points.y[i], points.t[i]);
        right_ids.push_back(ids[i]);
      }
    }
  }

  if (left_points.size() <= max_leaf_size) {
    LeafNode *left_leaf = new LeafNode(left_points, left_ids);
    this->left_leaf = left_leaf;
  } else {
    KDNode *left_kd = new KDNode(next_dim, left_points, left_ids);
    this->left_kd = left_kd;
  }

  if (right_points.size() <= max_leaf_size) {
    LeafNode *right_leaf = new LeafNode(right_points, right_ids);
    this->right_leaf = right_leaf;
  } else {
    KDNode *right_kd = new KDNode(next_dim, right_points, right_ids);
    this->right_kd = right_kd;
  }
}

KDNode::~KDNode() {
  if (this->left_kd != nullptr) {
    delete this->left_kd;
  }
  if (this->right_kd != nullptr) {
    delete this->right_kd;
  }
  if (this->left_leaf != nullptr) {
    delete this->left_leaf;
  }
  if (this->right_leaf != nullptr) {
    delete this->right_leaf;
  }
}

IDPoints KDNode::range_query(float x_min, float x_max, float y_min, float y_max) {
  if (this->dim == SplitDimension::X) {
    if (x_min <= this->split && this->split <= x_max) {
      // Straddle
      IDPoints left_points = this->left_leaf == nullptr ? this->left_kd->range_query(x_min, this->split, y_min, y_max)
                                                        : this->left_leaf->range_query_x(x_min, this->split);
      IDPoints right_points = this->right_leaf == nullptr
                                  ? this->right_kd->range_query(this->split, x_max, y_min, y_max)
                                  : this->right_leaf->range_query_x(this->split, x_max);
      return IDPoints(left_points, right_points);

    } else if (x_max < this->split) {
      // Query left
      if (this->left_leaf != nullptr) {
        return this->left_leaf->range_query_x(x_min, x_max);
      } else {
        return this->left_kd->range_query(x_min, x_max, y_min, y_max);
      }
    } else {
      // Query right
      if (this->right_leaf != nullptr) {
        return this->right_leaf->range_query_x(x_min, x_max);
      } else {
        return this->right_kd->range_query(x_min, x_max, y_min, y_max);
      }
    }
  } else {
    if (y_min <= this->split && this->split <= y_max) {
      // Straddle
      IDPoints left_points = this->left_leaf == nullptr ? this->left_kd->range_query(x_min, x_max, y_min, this->split)
                                                        : this->left_leaf->range_query_y(y_min, this->split);

      IDPoints right_points = this->right_leaf == nullptr
                                  ? this->right_kd->range_query(x_min, x_max, this->split, y_max)
                                  : this->right_leaf->range_query_y(this->split, y_max);
      return IDPoints(left_points, right_points);

    } else if (y_max < this->split) {
      // Query left
      if (this->left_leaf != nullptr) {
        return this->left_leaf->range_query_y(y_min, y_max);
      } else {
        return this->left_kd->range_query(x_min, x_max, y_min, y_max);
      }
    } else {
      // Query right
      if (this->right_leaf != nullptr) {
        return this->right_leaf->range_query_y(y_min, y_max);
      } else {
        return this->right_kd->range_query(x_min, x_max, y_min, y_max);
      }
    }
  }
}

namespace thoracuda {
  namespace kdtree {

    KDNode build_kdtree(GnomonicPointSources points, int max_leaf_size) {
      std::vector<int> ids;
      for (int i = 0; i < points.size(); i++) {
        ids.push_back(i);
      }
      return KDNode(SplitDimension::X, points, ids, max_leaf_size);
    }

    float median(std::vector<double> vals) {
      if (vals.size() == 0) {
        throw std::invalid_argument("median: vals is empty");
      }

      std::vector<double> vals_copy = vals;

      std::sort(vals_copy.begin(), vals_copy.end());
      if (vals_copy.size() % 2 == 0) {
        return (vals_copy[vals_copy.size() / 2 - 1] + vals_copy[vals_copy.size() / 2]) / 2;
      } else {
        return vals_copy[vals_copy.size() / 2];
      }
    }

  }  // namespace kdtree
}  // namespace thoracuda
