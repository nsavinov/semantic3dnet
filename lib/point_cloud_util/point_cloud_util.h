#ifndef __POINT_CLOUD_UTIL_H__
#define __POINT_CLOUD_UTIL_H__

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <limits>
#include <random>
#include <cmath>
#include <unordered_map>
#include <omp.h>
#include "voxel_grid3D.h"

#define XYZL float x, y, z; int label

using std::vector;
using std::unordered_map;
using std::cout;
using std::string;
using std::ifstream;
using std::ofstream;
using std::stringstream;
using std::endl;
using std::cerr;

#include "data_loader_constants.h"
const string kAggregatedSuffix = "_aggregated.txt";
const bool kIsExtended = (kWindowSize % 2 == 0);

struct DataLoader {
  DataLoader();
  void deferred_construct(const string& input_path,
                          int seed,
                          int number_of_scales,
                          float additional_rotation);
  DataLoader(const string& input_path,
             int seed = kDefaultSeed,
             int number_of_scales = kDefaultNumberOfScales,
             float additional_rotation = 0);
  ~DataLoader();
  void rotate_coordinates_around_z(float& x,
                                   float& y,
                                   float angle_in_radians);
  int expanded_to_global(int expanded, int center);
  int odd_local_to_global(int local, int center);
  void set_global_voxel_grid();
  void set_angle(bool is_random);
  void set_counts();
  void begin_iterating_lines(bool is_cached);
  void end_iterating_lines(bool is_cached);
  // true if success
  bool next_point(float& x, float& y, float& z, int& label, bool is_cached);
  void aggregate_in_volume(int number_of_augmentations, float sampling_rate);
  void set_up_streaming(bool is_random_angle);
  vector<float> get_aggregated_sample(float x, float y, float z, int label);
  void set_aggregated_sample_in_batch(float x, float y, float z, int label, int batch_position = -1);
  vector<float> get_class_counts();
  void filter_unlabelled();
  int get_number_of_points();
  // void transform_to_colored_ply();
  // static void print_encodings(const string& path, int number_of_scales);
  void cache_data();
  static int get_fixed_label(int label);
  static void merge_points_labels(const string& points_path,
                                  const string& labels_path);
  void set_class_shares();
  void set_up_infinite_streaming();
  void set_class_sampling_intervals();
  void set_up_class_generators();
  int sample_class_index();
  int sample_from_class(int class_index);
  float *next_random_batch();
  float get_aggregated_value(int scale_index,
                              int zero_local,
                              int one_local,
                              int two_local,
                              const array<int, 3>& center);
  void set_up_test_streaming();
  float *next_testing_batch();

  string path_;
  float angle_;
  vector<cvg::VoxelGrid3D > global_voxel_grid_scales_;
  vector<unordered_map<size_t, int> > counts_scales_;
  std::mt19937 random_engine_;
  std::uniform_real_distribution<> uniform_distribution_;
  std::uniform_int_distribution<int> uniform_int_;
  ifstream input_stream_;
  string line_;
  vector<array<float, 3> > cached_points_;
  vector<int> cached_labels_;
  size_t cached_index_;
  float *current_batch_;
  size_t batch_position_;
  int batch_counter_;
  vector<float> class_shares_;
  vector<float> class_sampling_intervals_;
  vector<vector<int> > class_inverse_index_;
  vector<std::uniform_int_distribution<int> > uniform_index_per_class_;
  int number_of_scales_;
  float additional_rotation_;
  int features_size_;
};

struct DataLoaderArray {
  DataLoaderArray();
  DataLoaderArray(const string& input_path,
                  int seed = kDefaultSeed,
                  int number_of_scales = kDefaultNumberOfScales,
                  int number_of_rotations = kDefaultNumberOfRotations);
  ~DataLoaderArray();
  void set_up_infinite_streaming();
  float *next_random_batch();
  void set_up_test_streaming();
  float *next_testing_batch();
  void set_batch();
  void aggregate_in_volume(int number_of_samples);

  DataLoader *array_;
  float *all_rotations_batch_;
  int number_of_rotations_;
  int number_of_scales_;
  int number_of_features_;
  string path_;
};

#endif
