#include "box3D.h"

namespace cvg {
Box3D::Box3D() 
    : bounds_{{{0, 0, 0}, {1, 1, 1}}} {}
Box3D::Box3D(const array<float, 3>& bound_min,
             const array<float, 3>& bound_max)
    : bounds_{bound_min, bound_max} {
}
bool Box3D::is_inside_box(const array<float, 3>& position,
                          float epsilon) const {
  for (int dimension = 0; dimension < position.size(); ++dimension) {
    if (position[dimension] < (bounds_[0][dimension] - epsilon) ||
        position[dimension] > (bounds_[1][dimension] + epsilon)) {
      return false;
    }
  }
  return true;
}
void Box3D::get_voxel_resolution(int number_of_voxels, array<int, 3>& result) {
  float xz_ratio = (bounds_[1][0] - bounds_[0][0]) /
                    (bounds_[1][2] - bounds_[0][2]);
  float yz_ratio = (bounds_[1][1] - bounds_[0][1]) /
                    (bounds_[1][2] - bounds_[0][2]);
  int res_z = static_cast<int>(pow(number_of_voxels / (xz_ratio * yz_ratio), 1.0 / 3));
  int res_x = static_cast<int>(res_z * xz_ratio);
  int res_y = static_cast<int>(res_z * yz_ratio);
  result[0] = res_x;
  result[1] = res_y;
  result[2] = res_z;
}
void Box3D::get_box_center(array<float, 3>& result) {
  for (int dimension = 0; dimension < bounds_[0].size(); ++dimension) {
    result[dimension] = (bounds_[0][dimension] + bounds_[1][dimension]) / 2.0;
  }
}
}
