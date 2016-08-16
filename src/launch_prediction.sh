#!/usr/bin/env bash
export OMP_NUM_THREADS=5

export CUDA_VISIBLE_DEVICES=0
nohup th test_model.lua ../data/benchmark/MarketplaceFeldkirch_Station4_rgb_intensity-reduced_test.txt > nohup0.txt &

# export CUDA_VISIBLE_DEVICES=1
nohup th test_model.lua ../data/benchmark/StGallenCathedral_station6_rgb_intensity-reduced_test.txt > nohup1.txt &

# export CUDA_VISIBLE_DEVICES=2
nohup th test_model.lua ../data/benchmark/sg27_station10_rgb_intensity-reduced_test.txt > nohup2.txt &

# export CUDA_VISIBLE_DEVICES=3
nohup th test_model.lua ../data/benchmark/sg28_Station2_rgb_intensity-reduced_test.txt > nohup3.txt &

