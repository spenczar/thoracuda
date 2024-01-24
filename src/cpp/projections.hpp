#pragma once

#include <Eigen/Dense>

#include "cartesian_point_sources.hpp"
#include "gnomonic_point_sources.hpp"

using Eigen::Matrix3d;
using Eigen::Vector3d;

namespace thoracuda {
  namespace projections {
    /// @brief Rotation matrix for a rotation to align the z-axis with
    /// the angular momentum vector of the center object (ie, normal
    /// to the orbital plane).
    Eigen::Matrix3d r1_matrix(Eigen::Vector3d &center_pos, Eigen::Vector3d &center_vel);

    Eigen::Matrix3d r2_matrix(Eigen::Matrix3d &r1, Eigen::Vector3d &center_pos);

    /// @brief Rotation matrix for a gnomonic projection.
    /// The gnomonic projection is a projection onto a plane tangent
    /// to the center object. The center object is at the origin of
    /// the coordinate system. The plane is tangent to the center
    /// object at the point of closest approach of the target object.
    Matrix3d gnomonic_rotation_matrix(Vector3d &center_pos, Vector3d &center_vel);

    /// @brief Project CartesianPointSources onto a plane tangent to
    /// the center object.
    GnomonicPointSources gnomonic_projection(CartesianPointSources &cps, Vector3d &center_pos, Vector3d &center_vel);
  }  // namespace projections
}  // namespace thoracuda
