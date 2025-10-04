#!/bin/bash

# 设置训练环境变量
export CUDA_VISIBLE_DEVICES=0  # 设置使用的GPU

python sample_compare_for_fid.py