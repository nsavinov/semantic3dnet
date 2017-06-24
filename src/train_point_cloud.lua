require 'torch'
require 'cutorch'
require 'math'
require 'nn'
require 'gnuplot'
require 'image'
require 'optim'
require 'point_cloud_model_definition'
require 'point_cloud_constants'
require 'point_cloud_error_calculation'
require 'point_cloud_loader'
require 'loader_ffi'

torch.manualSeed(10000)
cutorch.manualSeed(10000)
local opt = define_constants()

local optim_method = optim.adadelta

local train = load_point_cloud('../data/benchmark/sg28_station4_intensity_rgb_train.txt_aggregated.txt', 1000, opt)
print(train.data:size())
print(train.labels:size())
local test = load_point_cloud('../data/benchmark/bildstein_station1_xyz_intensity_rgb_train.txt_aggregated.txt', 1000, opt)
print(test.data:size())
print(test.labels:size())
local model
local optim_state

if opt.kWarmStart then
  print('warm start!')
  optim_state = torch.load(opt.kOptimStateDumpName)
  model = torch.load(opt.kModelDumpName)
  set_up_loader(optim_state.epoch, opt)
else
  print('creating fresh new model!')
  optim_state = {}
  optim_state.epoch = 1
  set_up_loader(optim_state.epoch, opt)
  model = define_model(opt.kSide, opt.n_outputs, opt.number_of_filters, opt.kNumberOfScales, opt.kNumberOfRotations)
end
local criterion = nn.ClassNLLCriterion()
local criterion = criterion:cuda()

local parameters, gradParameters = model:getParameters()

local batch
local batch_feval = function(x)
  if x ~= parameters then
    parameters:copy(x)
  end
  local batch_inputs = batch.data:cuda()
  local batch_targets = batch.labels:cuda()
  gradParameters:zero()
  local batch_outputs = model:forward(batch_inputs)
  local batch_loss = criterion:forward(batch_outputs, batch_targets)
  local dloss_doutput = criterion:backward(batch_outputs, batch_targets)
  model:backward(batch_inputs, dloss_doutput)
  return batch_loss, gradParameters
end

optim_state.test_errors = {}
optim_state.train_errors = {}
torch.manualSeed(optim_state.epoch)
cutorch.manualSeed(optim_state.epoch)
local batch_counter = 1
set_up_infinite_streaming()
while true do
  batch = get_next_random_batch(opt)
  if batch_counter % opt.kLargePrintingInterval ~= 0 then
    print(batch_counter)
    batch_counter = batch_counter + 1
    local _, minibatch_loss = optim_method(batch_feval, parameters, optim_state)
    collectgarbage()
  else
    batch_counter = 1
    torch.save(opt.kModelDumpName, model)
    torch.save(opt.kOptimStateDumpName, optim_state)
    optim_state.epoch = optim_state.epoch + 1
    optim_state.train_errors[#optim_state.train_errors + 1] = calculate_error(model, train, opt)
    optim_state.test_errors[#optim_state.test_errors + 1] = calculate_error(model, test, opt)
    print(string.format("optim_state.epoch: %6s, train_error = %6.6f, test_error = %6.6f",
                        optim_state.epoch,
                        optim_state.train_errors[#optim_state.train_errors],
                        optim_state.test_errors[#optim_state.test_errors]))
  end
end
