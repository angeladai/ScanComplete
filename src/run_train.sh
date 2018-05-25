#!/bin/bash

GPU=0
BATCH_SIZE=8
BASE_DIR='./train'
# Fill in training data filepattern here.
DATA='data/vox19_dim32/train*.tfrecords'      # data for 19cm level
#DATA='data/vox5-9-19_dim32/train_*.tfrecords' # data for 9cm and 5cm levels
NUMBER_OF_STEPS=100000

# coarse level
IS_BASE_LEVEL=1
HIERARCHY_LEVEL=3
STORED_BLOCK_DIM=32
STORED_BLOCK_HEIGHT=16
BLOCK_DIM=32
BLOCK_HEIGHT=16
TRAIN_SAMPLES=0
VERSION=003

## mid level
#IS_BASE_LEVEL=0
#HIERARCHY_LEVEL=2
#STORED_BLOCK_DIM=32
#STORED_BLOCK_HEIGHT=32
#BLOCK_DIM=32
#BLOCK_HEIGHT=32
#TRAIN_SAMPLES=1
#VERSION=002

## hi level
#IS_BASE_LEVEL=0
#HIERARCHY_LEVEL=1
#STORED_BLOCK_DIM=64
#STORED_BLOCK_HEIGHT=64
#BLOCK_DIM=32
#BLOCK_HEIGHT=64
#TRAIN_SAMPLES=1
#VERSION=001

PREDICT_SEMANTICS=0  # set to 1 to predict semantics
WEIGHT_SEM=0.5

python train.py \
  --gpu="${GPU}" \
  --train_dir=${BASE_DIR}/train_v${VERSION} \
  --batch_size="${BATCH_SIZE}" \
  --data_filepattern="${DATA}" \
  --stored_dim_block="${STORED_BLOCK_DIM}" \
  --stored_height_block="${STORED_BLOCK_HEIGHT}" \
  --dim_block="${BLOCK_DIM}" \
  --height_block="${BLOCK_HEIGHT}" \
  --hierarchy_level="${HIERARCHY_LEVEL}" \
  --is_base_level="${IS_BASE_LEVEL}" \
  --predict_semantics="${PREDICT_SEMANTICS}" \
  --weight_semantic="${WEIGHT_SEM}" \
  --number_of_steps="${NUMBER_OF_STEPS}"
