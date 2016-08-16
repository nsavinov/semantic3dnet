#!/usr/bin/env bash
mkdir ../dump
export CUDA_VISIBLE_DEVICES=0
nohup th train_point_cloud.lua &
