#include "point_cloud_util.h"

int main(int argc, char **argv) {
  {
    DataLoaderArray data_loader_array("../data/benchmark/sg28_station4_intensity_rgb_train.txt");
    data_loader_array.aggregate_in_volume(1000);
  }
  {
    DataLoaderArray data_loader_array("../data/benchmark/bildstein_station1_xyz_intensity_rgb_train.txt");
    data_loader_array.aggregate_in_volume(1000);
  }
  return 0;
}
