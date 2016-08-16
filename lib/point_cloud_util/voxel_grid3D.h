#ifndef __VOXEL_GRID3D_H__
#define __VOXEL_GRID3D_H__

#include "box3D.h"

namespace cvg {
class VoxelGrid3D {
 public:
  VoxelGrid3D();
  VoxelGrid3D(const Box3D& bounding_box,
              const array<int, 3>& grid_sizes);
  bool is_inside_grid(const array<int, 3>& coordinates);
  size_t world_coordinates_to_voxel_index(const array<float, 3>& world_coordinates);
  size_t get_voxel_index(const array<int, 3>& coordinates);
  // [) ... []
  void get_grid_coordinates(const array<float, 3>& position, array<int, 3>& result);

  Box3D bounding_box_;
  Box3D voxel_sized_box_;
  array<int, 3> grid_sizes_;
};
}

#endif
