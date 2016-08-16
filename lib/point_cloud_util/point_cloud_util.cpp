#include "point_cloud_util.h"

DataLoader::DataLoader(const string& input_path,
                       int seed,
                       int number_of_scales,
                       float additional_rotation)
    : number_of_scales_(number_of_scales)
    , additional_rotation_(additional_rotation) {
  random_engine_ = std::mt19937(seed);
  path_ = input_path;
  features_size_ = kWindowSize * kWindowSize * kWindowSize * number_of_scales_;
  current_batch_ = new float[1 + (features_size_ + 1) * kBatchSize];
}
DataLoader::DataLoader() {
}
void DataLoader::deferred_construct(const string& input_path,
                                    int seed,
                                    int number_of_scales,
                                    float additional_rotation) {
  number_of_scales_ = number_of_scales;
  additional_rotation_ = additional_rotation;
  random_engine_ = std::mt19937(seed);
  path_ = input_path;
  features_size_ = kWindowSize * kWindowSize * kWindowSize * number_of_scales_;
  current_batch_ = new float[1 + (features_size_ + 1) * kBatchSize];
}
DataLoader::~DataLoader() {
  delete[] current_batch_;
}
void DataLoader::rotate_coordinates_around_z(float& x,
                                             float& y,
                                             float angle_in_radians) {
  float modified_angle = angle_in_radians + 2 * M_PI * additional_rotation_;
  float new_x = cos(modified_angle) * x - sin(modified_angle) * y;
  float new_y = sin(modified_angle) * x + cos(modified_angle) * y;
  x = new_x;
  y = new_y;
}
int DataLoader::expanded_to_global(int expanded, int center) {
  int result;
  if (expanded >= kWindowSize / 2) {
    result = expanded - 1;
  } else {
    result = expanded;
  }
  result += center + 1 - (kWindowSize / 2);
  return result;
}
int DataLoader::odd_local_to_global(int local, int center) {
  return center - (kWindowSize / 2) + local;
}
void DataLoader::set_global_voxel_grid() {
  global_voxel_grid_scales_.clear();
  float min_x = std::numeric_limits<float>::max();
  float min_y = std::numeric_limits<float>::max();
  float min_z = std::numeric_limits<float>::max();
  float max_x = std::numeric_limits<float>::lowest();
  float max_y = std::numeric_limits<float>::lowest();
  float max_z = std::numeric_limits<float>::lowest();
  XYZL;
  begin_iterating_lines(true);
  while(next_point(x, y, z, label, true)) {
    rotate_coordinates_around_z(x, y, angle_);
    min_x = min(x, min_x);
    min_y = min(y, min_y);
    min_z = min(z, min_z);
    max_x = max(x, max_x);
    max_y = max(y, max_y);
    max_z = max(z, max_z);
  }
  end_iterating_lines(true);
  if (kPointCloudVerbose) {
    cout.precision(std::numeric_limits<float>::digits10 + 1);
    cout << "min_x: " << min_x << endl;
    cout << "min_y: " << min_y << endl;
    cout << "min_z: " << min_z << endl;
    cout << "max_x: " << max_x << endl;
    cout << "max_y: " << max_y << endl;
    cout << "max_z: " << max_z << endl;
  }
  cvg::Box3D bounding_box = cvg::Box3D(array<float, 3>{min_x - kSpatialResolution,
                                            min_y - kSpatialResolution,
                                            min_z - kSpatialResolution},
                                       array<float, 3>{max_x + kSpatialResolution,
                                                      max_y + kSpatialResolution,
                                                      max_z + kSpatialResolution});
  if (kPointCloudVerbose) {
    cout << "found bounding box" << endl;
  }
  for (int scale_index = 0;
       scale_index < number_of_scales_;
       ++scale_index) {
    float effective_resolution = kSpatialResolution *
                                  pow(2.0, static_cast<float>(scale_index));
    int size_x = (bounding_box.bounds_[1][0] - bounding_box.bounds_[0][0]) /
                 effective_resolution;
    int size_y = (bounding_box.bounds_[1][1] - bounding_box.bounds_[0][1]) /
                 effective_resolution;
    int size_z = (bounding_box.bounds_[1][2] - bounding_box.bounds_[0][2]) /
                 effective_resolution;
    if (kPointCloudVerbose) {
      cout << "size_x: " << size_x << endl <<
              "size_y: " << size_y << endl <<
              "size_z: " << size_z << endl;
    }
    global_voxel_grid_scales_.push_back(
        cvg::VoxelGrid3D(bounding_box, array<int, 3>{size_x, size_y, size_z}));
  } 
}
void DataLoader::set_angle(bool is_random) {
  if (is_random) {
    angle_ = 2 * M_PI * uniform_distribution_(random_engine_);
  } else {
    angle_ = 0;
  }
}
void DataLoader::set_counts() {
  counts_scales_ = vector<unordered_map<size_t, int> >(number_of_scales_);
  XYZL;
  for (int scale_index = 0;
       scale_index < number_of_scales_;
       ++scale_index) {
    begin_iterating_lines(true);
    while(next_point(x, y, z, label, true)) {
      rotate_coordinates_around_z(x, y, angle_);
      size_t voxel_index = global_voxel_grid_scales_[scale_index].world_coordinates_to_voxel_index(
          array<float, 3>{x, y, z});
      ++counts_scales_[scale_index][voxel_index];
    }
    end_iterating_lines(true);
  }
  if (kPointCloudVerbose) {
    cout << "finished counts" << endl;
  }
}
void DataLoader::begin_iterating_lines(bool is_cached) {
  if (is_cached) {
    if (cached_labels_.empty()) {
      if (kPointCloudVerbose) {
        cout << "starting caching" << endl;
      }
      cache_data();
      if (kPointCloudVerbose) {
        cout << "caching done" << endl;
      }
    }
    cached_index_ = 0;
  } else {
    input_stream_.open(path_);
  }
}
void DataLoader::end_iterating_lines(bool is_cached) {
  if (!is_cached) {
    input_stream_.close();
  }
}
// true if success
bool DataLoader::next_point(float& x, float& y, float& z, int& label, bool is_cached) {
  if (!is_cached) {
    if (!getline(input_stream_, line_)) {
      return false;
    } else {
      stringstream sstream(line_);
      sstream >> x >> y >> z >> label;
      return true;
    }
  } else {
    if (cached_index_ >= cached_labels_.size()) {
      return false;
    } else {
      x = cached_points_[cached_index_][0];
      y = cached_points_[cached_index_][1];
      z = cached_points_[cached_index_][2];
      label = cached_labels_[cached_index_];
      ++cached_index_;
      return true;
    }
  }
}
void DataLoader::aggregate_in_volume(int number_of_augmentations,
                                     float sampling_rate) {
  set_class_shares();
  ofstream output;
  output.open(path_ + kAggregatedSuffix);
  output.precision(std::numeric_limits<float>::digits10 + 1);
  for (int augmentation = 0; augmentation < number_of_augmentations; ++augmentation) {
    set_up_streaming(number_of_augmentations > 1);
    XYZL;
    begin_iterating_lines(true);
    while (next_point(x, y, z, label, true)) {
      if (label > 0) {
        if (uniform_distribution_(random_engine_) > sampling_rate / (kNumberOfClasses *
                                                                     class_shares_[label - 1])) {
          continue;
        }
        vector<float> current_sample = get_aggregated_sample(x, y, z, label);
        for (int index = 0; index < current_sample.size(); ++index) {
          output << current_sample[index] << ' ';
        }
        output << endl;
      }
    }
    end_iterating_lines(true);
  }
  output.close();
}
void DataLoader::set_up_streaming(bool is_random_angle) {
  set_angle(is_random_angle);
  set_global_voxel_grid();
  set_counts();
}
void DataLoader::set_class_shares() {
  class_shares_ = get_class_counts();
  for (int index = 0; index < class_shares_.size(); ++index) {
    if (class_shares_[index] == 0) {
      cerr << "Error: class " << index + 1 << " is missing from the training data." << endl;
      exit(-1);
    }
  }
  float sum = 0;
  for (float count : class_shares_) {
    sum += count;
  }
  for (float& count : class_shares_) {
    count /= sum;
  }
}
void DataLoader::set_up_test_streaming() {
  set_up_streaming(false);
  begin_iterating_lines(true);
}
float *DataLoader::next_testing_batch() {
  int real_batch_size = min(cached_labels_.size() - cached_index_,
                            static_cast<size_t>(kBatchSize));
  #pragma omp parallel for
  for (size_t sample_index = cached_index_;
       sample_index < cached_index_ + real_batch_size;
       ++sample_index) {
    XYZL;
    x = cached_points_[sample_index][0];
    y = cached_points_[sample_index][1];
    z = cached_points_[sample_index][2];
    label = cached_labels_[sample_index];
    int batch_position = 1 + (features_size_ + 1) *
                             (sample_index - cached_index_);
    set_aggregated_sample_in_batch(x, y, z, label, batch_position);
  }
  cached_index_ += real_batch_size;
  current_batch_[0] = static_cast<float>(real_batch_size);
  return current_batch_;
}
void DataLoader::set_up_infinite_streaming() {
  set_class_sampling_intervals();
  batch_counter_ = 0;
  set_up_streaming(true);
  set_up_class_generators(); 
}
void DataLoader::set_class_sampling_intervals() {
  class_sampling_intervals_ = vector<float>(kNumberOfClasses);
  for (int class_index = 0; class_index < kNumberOfClasses; ++class_index) {
    class_sampling_intervals_[class_index] = static_cast<float>(class_index + 1) /
                                             kNumberOfClasses;
  }
  class_sampling_intervals_.back() = 1;
}
void DataLoader::set_up_class_generators() {
  class_inverse_index_ = vector<vector<int> >(kNumberOfClasses, vector<int>());
  XYZL;
  begin_iterating_lines(true);
  int point_index = 0;
  while(next_point(x, y, z, label, true)) {
    if (label > 0) {
      class_inverse_index_[label - 1].push_back(point_index);
    }
    ++point_index;
  }
  end_iterating_lines(true);
  uniform_index_per_class_ = vector<std::uniform_int_distribution<int> >();
  for (int index = 0; index < kNumberOfClasses; ++index) {
    uniform_index_per_class_.push_back(
        std::uniform_int_distribution<int>(
            0,
            static_cast<int>(class_inverse_index_[index].size()) - 1));
  }
}
int DataLoader::sample_class_index() {
  float random_in_01 = uniform_distribution_(random_engine_);
  for (int class_index = 0; class_index < kNumberOfClasses; ++class_index) {
    if (random_in_01 <= class_sampling_intervals_[class_index]) {
      return class_index;
    }
  }
}
int DataLoader::sample_from_class(int class_index) {
  return class_inverse_index_[class_index]
                             [uniform_index_per_class_[class_index](random_engine_)];
}
float *DataLoader::next_random_batch() {
  if (batch_counter_ > kBatchResamplingLimit) {
    batch_counter_ = 0;
    set_up_streaming(true);
  }
  int sample_index = 0;
  batch_position_ = 1;
  while (sample_index < kBatchSize) {
    int random_class = sample_class_index();
    int random_choice = sample_from_class(random_class);
    ++sample_index;
    set_aggregated_sample_in_batch(cached_points_[random_choice][0],
                                   cached_points_[random_choice][1],
                                   cached_points_[random_choice][2],
                                   cached_labels_[random_choice]);
  }
  current_batch_[0] = static_cast<float>(sample_index);
  ++batch_counter_;
  return current_batch_;
}
vector<float> DataLoader::get_aggregated_sample(float x, float y, float z, int label) {
  vector<float> current_sample;
  rotate_coordinates_around_z(x, y, angle_);
  for (int scale_index = 0;
       scale_index < number_of_scales_;
       ++scale_index) {
    array<int, 3> center;
    global_voxel_grid_scales_[scale_index].get_grid_coordinates(
        array<float, 3>{x, y, z}, center);
    for (int zero_local = 0; zero_local < kWindowSize; ++zero_local) {
      for(int one_local = 0; one_local < kWindowSize; ++one_local) {
        for (int two_local = 0; two_local < kWindowSize; ++two_local) {
          current_sample.push_back(get_aggregated_value(scale_index,
                                                        zero_local,
                                                        one_local,
                                                        two_local,
                                                        center));
        }
      }
    }
  }
  current_sample.push_back(label);
  return current_sample;
}
float DataLoader::get_aggregated_value(int scale_index,
                                        int zero_local,
                                        int one_local,
                                        int two_local,
                                        const array<int, 3>& center) {
  int zero_global;
  int one_global;
  int two_global;
  if (kIsExtended) {
    zero_global = expanded_to_global(zero_local, center[0]);
    one_global = expanded_to_global(one_local, center[1]);
    two_global = expanded_to_global(two_local, center[2]);
  } else {
    zero_global = odd_local_to_global(zero_local, center[0]);
    one_global = odd_local_to_global(one_local, center[1]);
    two_global = odd_local_to_global(two_local, center[2]);
  }
  if (!global_voxel_grid_scales_[scale_index].is_inside_grid(array<int, 3>{zero_global,
                                                                         one_global,
                                                                         two_global})) {
    return static_cast<float>(0);
  } else {
    float value = 0;
    auto iterator = counts_scales_[scale_index].find(
        global_voxel_grid_scales_[scale_index].get_voxel_index(array<int, 3>{zero_global,
                                                                           one_global,
                                                                           two_global}));
    if (iterator != counts_scales_[scale_index].end()) {
      value = iterator->second;
    }
    return value;
  }
}
void DataLoader::set_aggregated_sample_in_batch(float x,
                                                float y,
                                                float z,
                                                int label,
                                                int batch_position) {
  bool synchronize = false;
  if (batch_position < 0) {
    batch_position = batch_position_;
    synchronize = true;
  }
  rotate_coordinates_around_z(x, y, angle_);
  for (int scale_index = 0;
       scale_index < number_of_scales_;
       ++scale_index) {
    array<int, 3> center;
    global_voxel_grid_scales_[scale_index].get_grid_coordinates(
        array<float, 3>{x, y, z},
        center);
    for (int zero_local = 0; zero_local < kWindowSize; ++zero_local) {
      for(int one_local = 0; one_local < kWindowSize; ++one_local) {
        for (int two_local = 0; two_local < kWindowSize; ++two_local) {
          current_batch_[batch_position] = get_aggregated_value(scale_index,
                                                                 zero_local,
                                                                 one_local,
                                                                 two_local,
                                                                 center);
          ++batch_position;
        }
      }
    }
  }
  current_batch_[batch_position] = static_cast<float>(label);
  ++batch_position;
  if (synchronize) {
    batch_position_ = batch_position;
  }
}
vector<float> DataLoader::get_class_counts() {
  vector<float> class_counts(kNumberOfClasses, 0);
  XYZL;
  begin_iterating_lines(true);
  while(next_point(x, y, z, label, true)) {
    if (label > 0) {
      ++class_counts[label - 1];
    }
  }
  end_iterating_lines(true);
  return class_counts;
}
int DataLoader::get_number_of_points() {
  int number_of_points = 0;
  XYZL;
  begin_iterating_lines(true);
  while (next_point(x, y, z, label, true)) {
    ++number_of_points;
  }
  end_iterating_lines(true);
  return number_of_points;
}
void DataLoader::cache_data() {
  XYZL;
  begin_iterating_lines(false);
  while (next_point(x, y, z, label, false)) {
    cached_points_.push_back(array<float, 3>{x, y, z});
    cached_labels_.push_back(label);
  } 
  end_iterating_lines(false);
}

DataLoaderArray::DataLoaderArray() {
}
DataLoaderArray::DataLoaderArray(const string& input_path,
                                 int seed,
                                 int number_of_scales,
                                 int number_of_rotations)
    : path_(input_path)
    , number_of_scales_(number_of_scales)
    , number_of_rotations_(number_of_rotations) {
  all_rotations_batch_ = new float[1 + (kWindowSize * kWindowSize * kWindowSize *
                                    number_of_scales_ * number_of_rotations_ + 1) * kBatchSize];
  array_ = new DataLoader[number_of_rotations_];
  for (int rotation_index = 0; rotation_index < number_of_rotations_; ++rotation_index) {
    array_[rotation_index].deferred_construct(
        input_path,
        seed,
        number_of_scales,
        static_cast<float>(rotation_index) / number_of_rotations_);
  }
  number_of_features_ = array_[0].features_size_;
}
DataLoaderArray::~DataLoaderArray() {
  delete[] all_rotations_batch_;
  delete[] array_;
}
void DataLoaderArray::set_up_infinite_streaming() {
  for (int rotation_index = 0; rotation_index < number_of_rotations_; ++rotation_index) {
    array_[rotation_index].set_up_infinite_streaming();
  }
}
void DataLoaderArray::set_up_test_streaming() {
  for (int rotation_index = 0; rotation_index < number_of_rotations_; ++rotation_index) {
    array_[rotation_index].set_up_test_streaming();
  }
}
void DataLoaderArray::set_batch() {
  int batch_size = array_[0].current_batch_[0];
  all_rotations_batch_[0] = batch_size;
  for (int rotation_index = 0; rotation_index < number_of_rotations_; ++rotation_index) {
    for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
      for (int feature_index = 0; feature_index < number_of_features_; ++feature_index) {
        all_rotations_batch_[1 +
                             batch_index * (number_of_features_ * number_of_rotations_ + 1) +
                             rotation_index * number_of_features_ +
                             feature_index] =
            array_[rotation_index].current_batch_[1 +
                                                  batch_index * (number_of_features_ + 1) +
                                                  feature_index];
      }
    }
  }
  for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
    all_rotations_batch_[1 +
                         batch_index * (number_of_features_ * number_of_rotations_ + 1) +
                         number_of_features_ * number_of_rotations_] =
        array_[0].current_batch_[1 +
                                 batch_index * (number_of_features_ + 1) +
                                 number_of_features_];
  }
} 
float *DataLoaderArray::next_random_batch() {
  for (int rotation_index = 0; rotation_index < number_of_rotations_; ++rotation_index) {
    array_[rotation_index].next_random_batch();
  }
  if (number_of_rotations_ == 1) {
    return array_[0].current_batch_;
  } else {
    set_batch();
    return all_rotations_batch_;
  }
}
float *DataLoaderArray::next_testing_batch() {
  for (int rotation_index = 0; rotation_index < number_of_rotations_; ++rotation_index) {
    array_[rotation_index].next_testing_batch();
  }
  if (number_of_rotations_ == 1) {
    return array_[0].current_batch_;
  } else {
    set_batch();
    return all_rotations_batch_;
  }
}
void DataLoaderArray::aggregate_in_volume(int number_of_samples) {
  int number_of_batches = number_of_samples / kBatchSize;
  ofstream output;
  output.open(path_ + kAggregatedSuffix);
  output.precision(std::numeric_limits<float>::digits10 + 1);
  set_up_infinite_streaming();
  for (int batch_index = 0; batch_index < number_of_batches; ++batch_index) {
    float *all_rotations_batch = next_random_batch();
    int batch_size = all_rotations_batch[0];
    for (int sample_index = 0; sample_index < batch_size; ++sample_index) {
      for (int index = 0; index < number_of_features_ * number_of_rotations_ + 1; ++index) {
        output << all_rotations_batch[1 + sample_index * (number_of_features_ * number_of_rotations_ + 1) + index] << ' ';
      }
      output << endl;
    }
  }
  output.close();
}
