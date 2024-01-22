#include "projections.hpp"

using Eigen::Matrix3d;
using Eigen::Vector3d;

namespace thoracuda {
  namespace projections {
    
    /// @brief Rotation matrix for a rotation to align the z-axis with
    /// the angular momentum vector of the center object (ie, normal
    /// to the orbital plane).
    Matrix3d r1_matrix(Vector3d &center_pos, Vector3d &center_vel) {
      // Compute a vector normal to the orbital plane of the center
      // object
      Vector3d n_hat = center_pos.cross(center_vel);
      // If center_vel is parallel to center_pos, then the cross
      // product is zero. In this case, we can choose any vector
      // normal to center_vel as the normal vector.
      if (n_hat.squaredNorm() == 0) {
        n_hat << 0.0, 1.0, 0.0;
      }
      n_hat.normalize();

      // Find the rotation axis, nu, which is the cross product of the
      // normal vector and the z-axis.
      Vector3d z_axis = {0.0, 0.0, 1.0};
      Vector3d nu = n_hat.cross(z_axis);

      // If the rotation axis is parallel to the z-axis, then the
      // rotation angle is zero, and the rotation matrix is the
      // identity matrix.
      if (nu.squaredNorm() == 0) {
        return Matrix3d::Identity();
      }

      // Find the cosine of the rotation angle, theta.
      //
      // Since n_hat and z_axis are unit vectors, the dot product of
      // the two is the cosine of the angle between them.
      double cos_theta = n_hat.dot(z_axis);

      // Compute the skew-symmetric matrix of the rotation axis.
      Matrix3d nu_skew;
      nu_skew << 0, -nu(2), nu(1), nu(2), 0, -nu(0), -nu(1), nu(0), 0;

      // Compute the rotation matrix.

      Matrix3d nu_skew_squared = nu_skew * nu_skew;
      nu_skew_squared *= 1 / (1 + cos_theta);

      Matrix3d r1 = Matrix3d::Identity() + nu_skew + nu_skew_squared;
      return r1;
    }

    Matrix3d r2_matrix(Matrix3d &r1, Vector3d &center_pos) {
      // First, rotate the center position vector by r1.
      Vector3d center_pos_rotated = r1 * center_pos;

      Matrix3d r2;
      r2 << (center_pos_rotated(1), -center_pos_rotated(0), 0,
	     center_pos_rotated(0), center_pos_rotated(1), 0,
	     0, 0, 1);
      return r2;
    }

    Matrix3d gnomonic_rotation_matrix(Vector3d &center_pos, Vector3d &center_vel) {
      Matrix3d r1 = r1_matrix(center_pos, center_vel);
      Matrix3d r2 = r2_matrix(r1, center_pos);
      return r2 * r1;
    }

    
  }  // namespace projections
}  // namespace thoracuda
