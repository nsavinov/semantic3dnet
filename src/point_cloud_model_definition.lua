require 'nn'
require 'cunn'
require 'cudnn'

function define_convolutional_model(input_size, number_of_filters)
  local model = nn.Sequential()
  model:add(nn.VolumetricConvolution(1, number_of_filters, 3, 3, 3, 1, 1, 1, 1, 1, 1))
  model:add(nn.ReLU())
  model:add(nn.VolumetricMaxPooling(2, 2, 2, 2, 2, 2))
  model:add(nn.VolumetricConvolution(number_of_filters, 2 * number_of_filters, 3, 3, 3, 1, 1, 1, 1, 1, 1))
  model:add(nn.ReLU())
  model:add(nn.VolumetricMaxPooling(2, 2, 2, 2, 2, 2))
  model:add(nn.VolumetricConvolution(2 * number_of_filters, 4 * number_of_filters, 3, 3, 3, 1, 1, 1, 1, 1, 1))
  model:add(nn.ReLU())
  model:add(nn.VolumetricMaxPooling(2, 2, 2, 2, 2, 2))
  model:add(nn.Reshape(4 * number_of_filters * ((input_size / 8) ^ 3)))
  return model
end

function define_model(input_size, n_outputs, number_of_filters, number_of_scales, number_of_rotations)
  local convolutional_model = define_convolutional_model(input_size, number_of_filters) 
  local scale_parallel_model = nn.Parallel(2, 2)
  for scale_index = 1, number_of_scales do
    scale_parallel_model:add(convolutional_model:clone())
  end
  local model_for_single_rotation = nn.Sequential()
  model_for_single_rotation:add(scale_parallel_model)
  model_for_single_rotation:add(nn.Reshape(1, 4 * number_of_filters * number_of_scales * ((input_size / 8) ^ 3)))
  local rotation_parallel_model = nn.Parallel(2, 2)
  for rotation_index = 1, number_of_rotations do
    rotation_parallel_model:add(model_for_single_rotation:clone())
  end
  local full_model = nn.Sequential()
  full_model:add(rotation_parallel_model)
  full_model:add(nn.SpatialMaxPooling(1, number_of_rotations))
  full_model:add(nn.Reshape(4 * number_of_filters * number_of_scales * ((input_size / 8) ^ 3)))
  local kFullyConnectedMultiplier = 128
  full_model:add(nn.Linear(4 * number_of_filters * number_of_scales * ((input_size / 8) ^ 3), kFullyConnectedMultiplier * number_of_filters))
  full_model:add(nn.ReLU())
  full_model:add(nn.Dropout())
  full_model:add(nn.Linear(kFullyConnectedMultiplier * number_of_filters, n_outputs))
  full_model:add(nn.LogSoftMax())
  full_model = full_model:cuda()
  full_model = cudnn.convert(full_model, cudnn)
  -- sharing parameters
  for rotation_index = 2, number_of_rotations do
    local current_module = full_model:get(1):get(rotation_index)
    current_module:share(full_model:get(1):get(1), 'weight', 'bias', 'gradWeight', 'gradBias')
  end
  full_model:training()
  print(full_model)
  return full_model
end
