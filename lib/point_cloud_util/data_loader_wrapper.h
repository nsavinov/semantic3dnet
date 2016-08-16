#ifndef __POINT_CLOUD_DECLARATIONS_H__
#define __POINT_CLOUD_DECLARATIONS_H__

extern "C" {
struct DataLoaderArray;
typedef struct DataLoaderWrapper {
  struct DataLoaderArray *inner_data_loader_ptr_;
} DataLoaderWrapper;
void construct(DataLoaderWrapper *self, int seed, const char *input_path, int number_of_scales, int number_of_rotations);
void destruct(DataLoaderWrapper *self);
void set_up_infinite_streaming(DataLoaderWrapper *self);
float *next_random_batch(DataLoaderWrapper *self);
int get_window_size();
void set_up_test_streaming(DataLoaderWrapper *self);
float *next_testing_batch(DataLoaderWrapper *self);
}
#endif
