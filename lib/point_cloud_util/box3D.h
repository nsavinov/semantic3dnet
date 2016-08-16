#ifndef __BOX3D_H__
#define __BOX3D_H__

#include <array>
#include <utility>
#include <algorithm>
#include <limits>
#include <cmath>
#include <iostream>

using std::pair;
using std::make_pair;
using std::for_each;
using std::min_element;
using std::distance;
using std::find;
using std::numeric_limits;
using std::cout;
using std::endl;
using std::array;
using std::min;
using std::max;

namespace cvg {
struct Box3D {
  Box3D();
  Box3D(const array<float, 3>& bound_min,
        const array<float, 3>& bound_max);
  bool is_inside_box(const array<float, 3>& position,
                     float epsilon = 0) const;
  void get_voxel_resolution(int number_of_voxels, array<int, 3>& result);
  void get_box_center(array<float, 3>& result);

  array<array<float, 3>, 2> bounds_;
};
}

#endif
