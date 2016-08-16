function load_cpp_options(file_name)
  local cpp_options = {}
  for line in io.lines(file_name) do
    local chunks = {}
    local equality_position = nil
    local counter = 1
    for w in line:gmatch("%S+") do
      chunks[#chunks + 1] = w
      if w == '=' then
        equality_position = counter
      end
      counter = counter + 1
    end
    if string.sub(chunks[equality_position + 1], -1, -1) == ';' then
      chunks[equality_position + 1] = string.sub(chunks[equality_position + 1], 1, -2)
    end
    cpp_options[chunks[equality_position - 1]] = tonumber(chunks[equality_position + 1])
  end
  return cpp_options
end

function define_constants()
  torch.setdefaulttensortype('torch.FloatTensor')
  local opt = {}
  local cpp_options = load_cpp_options('../lib/point_cloud_util/data_loader_constants.h')
  opt.kSide = cpp_options.kWindowSize
  opt.batch_size = cpp_options.kBatchSize
  opt.n_outputs = cpp_options.kNumberOfClasses
  opt.kNumberOfScales = cpp_options.kDefaultNumberOfScales
  opt.kNumberOfRotations = cpp_options.kDefaultNumberOfRotations
  opt.kSmallPrintingInterval = 2
  opt.number_of_filters = 16
  opt.kLargePrintingInterval = 100
  opt.kWarmStart = false
  opt.kModelDumpName = '../dump/model_dump'
  opt.kOptimStateDumpName = '../dump/optim_state_dump'
  opt.kStreamingPath = '../data/benchmark/sg28_station4_intensity_rgb_train.txt'
  return opt
end
