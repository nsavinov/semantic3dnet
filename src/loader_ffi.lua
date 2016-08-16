require 'point_cloud_constants'
local opt = define_constants()
ffi = require 'ffi'

ffi.cdef([[
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
]])
loader_lib = ffi.load('../build/lib/point_cloud_util/libpointCloudUtil.so')
kStreamingPath = opt.kStreamingPath
global_regular_batch = torch.FloatTensor((opt.kNumberOfRotations * opt.kNumberOfScales * (opt.kSide ^ 3) + 1) * opt.batch_size)

function set_up_loader(loader_seed, opt)
  LoaderType = ffi.typeof('DataLoaderWrapper')
  loader = LoaderType()
  print("loader seed: ", loader_seed)
  loader_lib.construct(loader, loader_seed, kStreamingPath, opt.kNumberOfScales, opt.kNumberOfRotations)
end

function set_up_testing_loader(testing_path, opt)
  LoaderType = ffi.typeof('DataLoaderWrapper')
  loader = LoaderType()
  loader_lib.construct(loader, 1, testing_path, opt.kNumberOfScales, opt.kNumberOfRotations)
end

function set_up_infinite_streaming()
  loader_lib.set_up_infinite_streaming(loader)
end

function set_up_test_streaming()
  loader_lib.set_up_test_streaming(loader)
end

function get_window_size()
  return loader_lib.get_window_size()
end

function get_next_random_batch(opt)
  local pointer = loader_lib.next_random_batch(loader)
  return transform(pointer, opt)
end

function transform(pointer, opt)
  local batch_size = pointer[0]
  local batch = nil
  if (batch_size ~= opt.batch_size) then
    batch = torch.FloatTensor((opt.kNumberOfRotations * opt.kNumberOfScales * (opt.kSide ^ 3) + 1) * batch_size)
  else
    batch = global_regular_batch
  end
  ffi.copy(batch:data(), pointer + 1, batch:nElement() * ffi.sizeof('float'))
  local loaded_data = batch:resize(batch_size, (opt.kNumberOfRotations * opt.kNumberOfScales * (opt.kSide ^ 3) + 1))
  local data = {}
  data.data = loaded_data[{{}, {1, -2}}]:clone()
  data.data = data.data:view(batch_size, opt.kNumberOfRotations, opt.kNumberOfScales, opt.kSide ^ 3)
  data.labels = loaded_data[{{}, {-1, -1}}]:clone():squeeze()
  data.data = torch.gt(data.data, 0)
  -- data.data:cdiv(data.data:max(4):expand(data.data:size()))
  data.data = data.data:view(batch_size, opt.kNumberOfRotations, opt.kNumberOfScales, 1, opt.kSide, opt.kSide, opt.kSide)
  return data
end

function get_next_testing_batch(opt)
  local pointer = loader_lib.next_testing_batch(loader)
  if (pointer[0] > 0) then
    return transform(pointer, opt)
  else
    return nil
  end
end
