#include "data_loader_wrapper.h"
#include "point_cloud_util.h"

void construct(DataLoaderWrapper *self, int seed, const char *input_path, int number_of_scales, int number_of_rotations) {
  self->inner_data_loader_ptr_ = new DataLoaderArray(string(input_path), seed, number_of_scales, number_of_rotations);
}
void destruct(DataLoaderWrapper *self) {
  delete self->inner_data_loader_ptr_;
}
void set_up_infinite_streaming(DataLoaderWrapper *self) {
  (self->inner_data_loader_ptr_)->set_up_infinite_streaming();
}
float *next_random_batch(DataLoaderWrapper *self) {
  return (self->inner_data_loader_ptr_)->next_random_batch();
}
int get_window_size() {
  return kWindowSize;
}
void set_up_test_streaming(DataLoaderWrapper *self) {
  (self->inner_data_loader_ptr_)->set_up_test_streaming();
}
float *next_testing_batch(DataLoaderWrapper *self) {
  return (self->inner_data_loader_ptr_)->next_testing_batch();
}
