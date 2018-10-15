#!/bin/bash

# Runs scan completion with hierarchical model over a set of scenes.

# Parameter section begins here. Edit to change number of test scenes, which model to use, output path.
MAX_NUM_TEST_SCENES=1000
NUM_HIERARCHY_LEVELS=1
BASE_OUTPUT_DIR=/home/github/ScanComplete/results_3frog

# Fill in path to test scenes
TEST_SCENES_PATH='/home/github/ScanComplete/data_3frog/vox19'

# Fill in model to use here
PREDICT_SEMANTICS=1
USED_MODEL='/home/github/ScanComplete/models/completion-semantics/hierarchy1of1/'

# Specify output folder
OUTPUT_FOLDER=${BASE_OUTPUT_DIR}/vis_level_single/

# Specify padding
# For now we consider a small 3 pixel padding in the voxel grid for 3frog
PADDING_SINGLE_LEVEL=2
HEIGHT_SINGLE_LEVEL=32

# End parameter section.

# Run hierarchy.
IS_BASE_LEVEL=1
HIERARCHY_LEVEL=1

# Go through all test scenes.
count=1
REC_COUNTER=0
for scene in $TEST_SCENES_PATH/*.tfrecords; do
  ((REC_COUNTER++))
done
TOTAL_SCENES=$(($REC_COUNTER<$MAX_NUM_TEST_SCENES?$REC_COUNTER:$MAX_NUM_TEST_SCENES))
for scene in $TEST_SCENES_PATH/*.tfrecords; do
  echo "Processing single precision model, scene $count of $TOTAL_SCENES: $scene".
  python complete_scan_3frog.py \
    --alsologtostderr \
    --base_dir="${USED_MODEL}" \
    --height_input="${HEIGHT_SINGLE_LEVEL}" \
    --hierarchy_level="${HIERARCHY_LEVEL}" \
    --num_total_hierarchy_levels="${NUM_HIERARCHY_LEVELS}" \
    --is_base_level="${IS_BASE_LEVEL}" \
    --predict_semantics="${PREDICT_SEMANTICS}" \
    --output_folder="${OUTPUT_FOLDER}" \
    --input_scene="${scene}" \
    --pad_test="${PADDING_SINGLE_LEVEL}"
  ((count++))
  if (( count > MAX_NUM_TEST_SCENES )); then
    break
  fi
done


