#!/bin/bash

GPU=0
BATCH_SIZE=8
BASE_DIR='./train'
# Fill in training data filepattern here.
DATA=''
NUMBER_OF_STEPS=100000

STORED_BLOCK_DIM_HI=64
STORED_BLOCK_HEIGHT_HI=64

IS_BASE_LEVEL=1
HIERARCHY_LEVEL=1
BLOCK_DIM=32
BLOCK_HEIGHT=64
PREDICT_SEMANTICS=0  # set to 1 to predict semantics
WEIGHT_SEM=0.5

VERSION=000

python train.py \
  --gpu="${GPU}" \
  --train_dir=${BASE_DIR}/train_v${VERSION} \
  --batch_size="${BATCH_SIZE}" \
  --data_filepattern="${DATA}" \
  --stored_dim_block_hi="${STORED_BLOCK_DIM_HI}" \
  --stored_height_block_hi="${STORED_BLOCK_HEIGHT_HI}" \
  --dim_block="${BLOCK_DIM}" \
  --height_block="${BLOCK_HEIGHT}" \
  --hierarchy_level="${HIERARCHY_LEVEL}" \
  --is_base_level="${IS_BASE_LEVEL}" \
  --predict_semantics="${PREDICT_SEMANTICS}" \
  --weight_semantic="${WEIGHT_SEM}" \
  --number_of_steps="${NUMBER_OF_STEPS}"
