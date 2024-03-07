#include "dbscan.hpp"

#include <map>
#include <set>
#include <vector>

#include "kdtree.hpp"

using Eigen::Vector2d;

using thoracuda::GnomonicPointSources;

namespace thoracuda {
namespace dbscan {

DBSCAN::DBSCAN(float eps, int min_size, GnomonicPointSources points) {
  this->eps = eps;
  this->min_size = min_size;
  this->points = points;
}

std::vector<Cluster> DBSCAN::fit() {
  thoracuda::kdtree::KDNode kd = thoracuda::kdtree::build_kdtree(this->points, 32);

  std::map<int, PointType> point_types;
  std::vector<Cluster> clusters;
  std::set<int> enqueued;
  thoracuda::kdtree::IDPoints neighbors;

  for (int i = 0; i < this->points.size(); i++) {
    if (point_types.find(i) != point_types.end()) {
      // Already visited
      continue;
    }
    enqueued.insert(i);
    neighbors = kd.range_query(this->points.x[i] - this->eps, this->points.x[i] + this->eps,
                               this->points.y[i] - this->eps, this->points.y[i] + this->eps);

    if (neighbors.size() < this->min_size) {
      // Too few neighbors
      point_types[i] = NOISE;
      continue;
    }

    for (int j = 0; j < neighbors.size(); j++) {
      enqueued.insert(neighbors.ids[j]);
    }

    // New cluster
    Cluster cluster;
    cluster.ids.reserve(neighbors.size());
    cluster.ids.push_back(i);
    point_types[i] = CORE;
    while (!neighbors.empty()) {
      thoracuda::kdtree::IDPoint p = neighbors.pop();
      auto search = point_types.find(p.id);
      if (search != point_types.end()) {
        // Already visited
        if (search->second == NOISE) {
          // Join the cluster
          point_types[p.id] = BORDER;
          cluster.ids.push_back(p.id);
        }
        continue;
      }

      // Join the cluster
      cluster.ids.push_back(p.id);
      thoracuda::kdtree::IDPoints new_neighbors =
          kd.range_query(p.x - this->eps, p.x + this->eps, p.y - this->eps, p.y + this->eps);
      if (new_neighbors.size() >= this->min_size) {
        point_types[p.id] = CORE;
        for (int j = 0; j < new_neighbors.size(); j++) {
          if (point_types.find(new_neighbors.ids[j]) == point_types.end()) {
            if (enqueued.count(new_neighbors.ids[j]) == 0) {
              enqueued.insert(new_neighbors.ids[j]);
              neighbors.push(thoracuda::kdtree::IDPoint(new_neighbors.ids[j], new_neighbors.points.x[j],
                                                        new_neighbors.points.y[j], new_neighbors.points.t[j]));
            }
          }
        }
      } else {
        point_types[p.id] = BORDER;
      }
    }

    clusters.push_back(cluster);
  }

  return clusters;
}

}  // namespace dbscan
}  // namespace thoracuda
