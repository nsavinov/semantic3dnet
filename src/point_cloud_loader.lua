function load_point_cloud(file_name, count, opt)
  local loaded_chunk = {}
  local loaded_tensor
  local i = 0
  for line in io.lines(file_name) do
    i = i + 1
    -- print(i)
    local chunks = {}
    for w in line:gmatch("%S+") do chunks[#chunks + 1] = tonumber(w) end
    loaded_chunk[#loaded_chunk + 1] = chunks
    if (i % 100 == 0 or i >= count) then
      if (loaded_tensor) then
        local temp = torch.cat(loaded_tensor, torch.Tensor(loaded_chunk), 1)
        loaded_tensor = temp
      else
        loaded_tensor = torch.Tensor(loaded_chunk)
      end
      loaded_chunk = {}
      if (i >= count) then
        break
      end
    end
  end
  local loaded_data = loaded_tensor
  count = loaded_data:size(1)
  local data = {}
  data.data = loaded_data[{{}, {1, -2}}]:clone()
  data.data = data.data:view(count, opt.kNumberOfRotations, opt.kNumberOfScales, opt.kSide * opt.kSide * opt.kSide)
  data.labels = loaded_data[{{}, {-1, -1}}]:clone():squeeze()
  data.data = torch.gt(data.data, 0)
  -- data.data:cdiv(data.data:max(4):expand(data.data:size()))
  data.data = data.data:view(count, opt.kNumberOfRotations, opt.kNumberOfScales, 1, opt.kSide, opt.kSide, opt.kSide)
  print('--------------------------------')
  print('inputs', data.data:size())
  print('targets', data.labels:size())
  print('min target', data.labels:min())
  print('max target', data.labels:max())
  print('--------------------------------')
  return data
end
