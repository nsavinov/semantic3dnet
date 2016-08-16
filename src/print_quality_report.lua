require 'point_cloud_constants'
require 'point_cloud_model_definition'
require 'point_cloud_loader'

local opt = define_constants()
torch.manualSeed(100)
cutorch.manualSeed(100)

local class_names = {'man made terrain',
                     'natural terrain',
                     'high vegetation',
                     'low vegetation',
                     'buildings',
                     'other hard scape',
                     'scanning artefacts'}

local test = load_point_cloud('/home/nsavinov/projects/deep_learning/point_cloud_labelling/PointClouds/LOD2Lables/StGallenCathedral/shuffled.txt', 1000, opt)
local model = torch.load(opt.kModelDumpName)
model:evaluate()
local data_size = test.data:size(1)
local batches_per_dataset = math.ceil(data_size / opt.batch_size)
local confusion_matrix = torch.Tensor(opt.n_outputs, opt.n_outputs):fill(0) --first real, second predicted
for batch_index = 0, (batches_per_dataset - 1) do
  local start_index = batch_index * opt.batch_size + 1
  local end_index = math.min(data_size, (batch_index + 1) * opt.batch_size)
  local batch_targets = test.labels[{{start_index, end_index}}]:cuda()
  local batch_inputs = test.data[{{start_index, end_index}}]:cuda()
  local logProbs = model:forward(batch_inputs)
  local classProbabilities = torch.exp(logProbs)
  local _, classPredictions = torch.max(classProbabilities, 2)
  classPredictions = classPredictions:squeeze()
  for index = 1, classPredictions:size(1) do
    confusion_matrix[batch_targets[index]][classPredictions[index]] =
        confusion_matrix[batch_targets[index]][classPredictions[index]] + 1
  end
  collectgarbage()
end
local eps = 0.0000001
local precisions = torch.diag(confusion_matrix):cdiv(torch.sum(confusion_matrix, 1) + eps)
local recalls = torch.diag(confusion_matrix):cdiv(torch.sum(confusion_matrix, 2) + eps)

print('legend', class_names)
print('confusion_matrix')
print(confusion_matrix)
print('precisions')
print(precisions)
print('recalls')
print(recalls)
