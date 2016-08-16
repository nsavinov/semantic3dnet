require 'point_cloud_constants'
require 'point_cloud_model_definition'
require 'loader_ffi'

function table_concat(t1, t2)
  for i = 1, #t2 do
    t1[#t1 + 1] = t2[i]
  end
  return t1
end

local opt = define_constants()
torch.manualSeed(100)
cutorch.manualSeed(100)
set_up_testing_loader(arg[1], opt)
local model = torch.load(opt.kModelDumpName)
model:evaluate()
set_up_test_streaming()
local batch_counter = 0
local all_predictions = {}
while (true) do
  if batch_counter % 1000 == 0 then
    print(batch_counter)
  end
  batch_counter = batch_counter + 1
  batch = get_next_testing_batch(opt)
  if (not batch) then
    break;
  end
  local batch_targets = batch.labels:cuda()
  local batch_inputs = batch.data:cuda()
  local logProbs = model:forward(batch_inputs)
  local classProbabilities = torch.exp(logProbs)
  local _, classPredictions = torch.max(classProbabilities, 2)
  local classPredictions = classPredictions:squeeze()
  classPredictions = torch.totable(classPredictions)
  table_concat(all_predictions, classPredictions)
  collectgarbage()
end
local result_file = io.open(arg[1] .. '_predictions.txt', "w")
for index = 1, #all_predictions do
  result_file:write(string.format("%d\n", all_predictions[index]))  
end

