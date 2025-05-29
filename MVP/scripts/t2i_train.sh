#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

torchrun --nproc_per_node=1 --master_port=29501 t2i_train.py \
  --depth=26 \
  --bs=16 \
  --ep=1 \
  --fp16=1 \
  --tlr=1e-4 \
  --alng=1e-3 \
  --wpe=0.1 \
  --control_strength=1.0 \
  --n_cond_embed=768 \
  --data_load_reso=256 \
  --outer_nums=20