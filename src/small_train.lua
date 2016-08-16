require 'torch'
require 'cutorch'
require 'math'
require 'nn'
require 'gnuplot'
require 'image'
require 'optim'
require 'point_cloud_model_definition'
require 'point_cloud_error_calculation'
require 'point_cloud_loader'
require 'point_cloud_constants'

torch.manualSeed(10000)
cutorch.manualSeed(10000)

local opt = define_constants()
local model = define_model(opt.kSide, opt.n_outputs, opt.number_of_filters, opt.kNumberOfScales, opt.kNumberOfRotations)
print(model)

local optim_method = optim.adadelta

local train = load_point_cloud('../data/benchmark/sg28_station4_intensity_rgb_train.txt_aggregated.txt', 1000, opt)
local test = load_point_cloud('../data/benchmark/bildstein_station1_xyz_intensity_rgb_train.txt_aggregated.txt', 1000, opt)
local optim_state = {}
optim_state.epoch = 1
local criterion = nn.ClassNLLCriterion()
local criterion = criterion:cuda()
criterion = cudnn.convert(criterion, cudnn)

local parameters, gradParameters = model:getParameters()
local batch = {}
local data_size = train.data:size(1)
local batches_per_dataset = math.ceil(data_size / opt.batch_size)

local batch_feval = function(x)
  if x ~= parameters then
    parameters:copy(x)
  end
  local batch_inputs = batch.data
  local batch_targets = batch.labels
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
while true do
  if optim_state.epoch % opt.kSmallPrintingInterval == 0 then
    optim_state.train_errors[#optim_state.train_errors + 1] = calculate_error(model, train, opt)
    optim_state.test_errors[#optim_state.test_errors + 1] = calculate_error(model, test, opt)
    print(string.format("optim_state.epoch: %6s, train_error = %6.6f, test_error = %6.6f",
                        optim_state.epoch,
                        optim_state.train_errors[#optim_state.train_errors],
                        optim_state.test_errors[#optim_state.test_errors]))
  end
  for batch_index = 0, (batches_per_dataset - 1) do
    local start_index = batch_index * opt.batch_size + 1
    local end_index = math.min(data_size, (batch_index + 1) * opt.batch_size)
    local batch_targets = train.labels[{{start_index, end_index}}]:cuda()
    local batch_inputs = train.data[{{start_index, end_index}}]:cuda()
    batch.data = batch_inputs
    batch.labels = batch_targets
    local _, minibatch_loss = optim_method(batch_feval, parameters, optim_state)
    collectgarbage()
  end
  optim_state.epoch = optim_state.epoch + 1
end
