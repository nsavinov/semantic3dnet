function calculate_error(model, data_to_check, opt)
  model:evaluate()
  local data_size = data_to_check.data:size(1)
  local batches_per_dataset = math.ceil(data_size / opt.batch_size)
  local error = 0
  for batch_index = 0, (batches_per_dataset - 1) do
    local start_index = batch_index * opt.batch_size + 1
    local end_index = math.min(data_size, (batch_index + 1) * opt.batch_size)
    local batch_targets = data_to_check.labels[{{start_index, end_index}}]:cuda()
    local batch_inputs = data_to_check.data[{{start_index, end_index}}]:cuda()
    local logProbs = model:forward(batch_inputs)
    local classProbabilities = torch.exp(logProbs)
    local _, classPredictions = torch.max(classProbabilities, 2)
    error = error + classPredictions:ne(batch_targets:typeAs(classPredictions)):sum()
    collectgarbage()
  end
  model:training()
  return error / data_size
end
