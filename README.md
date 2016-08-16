# Point cloud semantic segmentation via Deep 3D Convolutional Neural Network

This code implements a deep neural network for 3D point cloud semantic segmentation. It comes as a baseline for the benchmark http://www.semantic3d.net/ (reproduces DeepNet entry in reduced-8 track). It is written in C++/lua and is supposed to simplify starting to work with the benchmark.

The code requires at least 8 Gb RAM and an Nvidia GPU (at least 6 Gb of memory, tested for Nvidia Titan X GPU).

# How does it work
![](https://4.downloader.disk.yandex.com/disk/ebe2c6c112a20f065f31f21dfe92694ea2de98000d48d0f1d3c89e656c466132/57b355af/v42FHEdYgE4gy8O51xR16w4eyeXUn1bqN9q4TAWEw48_4ERB1QJCV_kd8YZ7_2_4YPbvU6cJ-5yATEuMHhUA4Q%3D%3D?uid=0&filename=Screen%20Shot%202016-08-16%20at%202.11.43%20PM.png&disposition=inline&hash=&limit=0&content_type=image%2Fpng&fsize=161730&hid=441b8712e979970098e9f71d3fdd6294&media_type=image&tknv=v2&etag=b4639d604635ed675840f21eae30d050)
Each point in the point cloud is to be classified into one of the semantic classes like building/car/vegetation/etc. It is done by considering a range of neighbourhoods for the point, computing occupancy grids for them and applying 3D convolutional neural network on those grids.
A more detailed description can be found at https://goo.gl/TUPqXo.

# Instructions for Linux (tested for Ubuntu 14.04.2 LTS):
1. install torch (tested for commit ba34004a7a48806bf4a6e88ecfc5fbada5efe636 from May 7, 2016) and cudnn (tested for v4). For the latter, follow the instructions https://github.com/soumith/cudnn.torch
2. clone this repository
3. run "cd build; ./setup.sh" to download data and transform it into necessary format.
4. run "./build_run.sh" to prepare small train/validation sets to track optimization progress
5. run "cd ../src; ./launch_training.sh". You can track the progress in nohup.out file. Wait until the train error becomes close to 0 and test error start to oscillate around some value. Then kill the process (took 304 epochs for the baseline).
6. run "./launch_prediction.sh". You might want to change gpu indexes in this script depending on number of gpus you have available. You can also tweak number of openmp threads. Wait until it finishes (might take a day or so).
7. after prediction finishes, run "./prepare_to_submit.sh" to put the submission into necessary format.
8. submit data/benchmark/submit.zip to the server http://www.semantic3d.net/submit_public.php.

# Parameters
The changeable constants are in lib/point_cloud_util/data_loader_constants.h and
in src/point_cloud_constants.lua

Explanation of them:

## lib/point_cloud_util/data_loader_constants.h:

const int kWindowSize = 16; // neighbourhood of the point is voxelized into kWindowSize ^ 3 voxels

const int kBatchSize = 100; // since batch is constructed on cpp side, we need specify its size here

const int kNumberOfClasses = 8; // integer labels 1, ..., 8 are considered

const int kDefaultNumberOfScales = 5; // each sample in the batch is constructed as stacked multiples scales. in the deep net architecture fully connected layer outputs are concatenated

const int kDefaultNumberOfRotations = 1; // optional implemenation of TI-pooling on top of multi-scale architecture. read the paper for more details

const float kSpatialResolution = 0.025; // voxel side in meters

const int kBatchResamplingLimit = 100; // after this number of batches, the coordinate system is rotated by a random angle and the voxelized representations are recalculated from scratch (augmentation).

## src/point_cloud_constants.lua:

opt.number_of_filters = 16 -- number of filters in the first convolutional layer, other layers size is also proportional to this constant.

opt.kLargePrintingInterval = 100 -- how often to evaluate model and dump solution to disk

opt.kWarmStart = false -- restart with the saved model

opt.kModelDumpName = '../dump/model_dump' -- where the model is saved

opt.kOptimStateDumpName = '../dump/optim_state_dump' -- where the optimization progress is saved

opt.kStreamingPath = '../data/benchmark/sg28_station4_intensity_rgb_train.txt' -- from which file training data is sampled

# Caveats
Data is randomly sampled from the training set, all the classes are made equally probable via this sampling. Thus it is required that the training file contains at least one sample of each class.
