require('loader_ffi')
require('point_cloud_constants')

opt = define_constants()
set_up_loader(1, opt)
set_up_infinite_streaming()
local batch_count = 0
while true do
  local batch = get_next_random_batch(opt)
  print(batch_count)
  batch_count = batch_count + 1  
end
