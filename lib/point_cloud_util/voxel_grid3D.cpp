#include "voxel_grid3D.h"

namespace cvg {
VoxelGrid3D::VoxelGrid3D() {}
VoxelGrid3D::VoxelGrid3D(const Box3D& bounding_box,
                         const array<int, 3>& grid_sizes)
    : bounding_box_(bounding_box)
    , grid_sizes_(grid_sizes) {
  array<float, 3> voxel_sizes;
  for (int dimension = 0; dimension < grid_sizes.size(); ++dimension) {
    voxel_sizes[dimension] = ((bounding_box_.bounds_[1][dimension] -
                           bounding_box_.bounds_[0][dimension]) / grid_sizes_[dimension]);
  }
  voxel_sized_box_ = Box3D(array<float, 3>{0, 0, 0}, voxel_sizes);
}
bool VoxelGrid3D::is_inside_grid(const array<int, 3>& coordinates) {
  for (int index = 0; index < grid_sizes_.size(); ++index) {
    if (coordinates[index] < 0 || coordinates[index] >= grid_sizes_[index]) {
      return false;
    }
  }
  return true;
}
size_t VoxelGrid3D::world_coordinates_to_voxel_index(const array<float, 3>& world_coordinates) {
  static array<int, 3> grid_coordinates;
  get_grid_coordinates(world_coordinates, grid_coordinates);
  return get_voxel_index(grid_coordinates);
}
size_t VoxelGrid3D::get_voxel_index(const array<int, 3>& coordinates) {
  return static_cast<size_t>(grid_sizes_[0]) *
           static_cast<size_t>(grid_sizes_[1]) *
           static_cast<size_t>(coordinates[2]) +
         static_cast<size_t>(grid_sizes_[0]) *
           static_cast<size_t>(coordinates[1]) +
         static_cast<size_t>(coordinates[0]);
}
void VoxelGrid3D::get_grid_coordinates(const array<float, 3>& position,
                                       array<int, 3>& result) {
  for (int dimension = 0; dimension < position.size(); ++dimension) {  
    result[dimension] = (min(static_cast<int>(grid_sizes_[dimension]) - 1,
                         static_cast<int>((position[dimension] - bounding_box_.bounds_[0][dimension]) /
                                          voxel_sized_box_.bounds_[1][dimension])));
  }
}
}
