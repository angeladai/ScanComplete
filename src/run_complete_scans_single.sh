#!/bin/bash

# Runs scan completion with hierarchical model over a set of scenes.

# Parameter section begins here. Edit to change number of test scenes, which model to use, output path.
MAX_NUM_TEST_SCENES=1
NUM_HIERARCHY_LEVELS=1
BASE_OUTPUT_DIR=/home/github/ScanComplete/results

# Fill in path to test scenes
TEST_SCENES_PATH='/home/github/ScanComplete/data/vox19'

# Fill in model to use here
PREDICT_SEMANTICS=1
USED_MODEL='/home/github/ScanComplete/models/completion-semantics/hierarchy1of1/'

# Specify output folder
OUTPUT_FOLDER=${BASE_OUTPUT_DIR}/vis_level_single/

# End parameter section.


# Run hierarchy.

IS_BASE_LEVEL=1
HIERARCHY_LEVEL=1
HEIGHT_INPUT=16

# Go through all test scenes.
count=1
for scene in $TEST_SCENES_PATH/*__0__.tfrecords; do
  echo "Processing hierarchy level 3, scene $count of $MAX_NUM_TEST_SCENES: $scene".
  python complete_scan.py \
    --alsologtostderr \
    --base_dir="${USED_MODEL}" \
    --height_input="${HEIGHT_INPUT}" \
    --hierarchy_level="${HIERARCHY_LEVEL}" \
    --num_total_hierarchy_levels="${NUM_HIERARCHY_LEVELS}" \
    --is_base_level="${IS_BASE_LEVEL}" \
    --predict_semantics="${PREDICT_SEMANTICS}" \
    --output_folder="${OUTPUT_FOLDER}" \
    --input_scene="${scene}"
  ((count++))
  if (( count > MAX_NUM_TEST_SCENES )); then
    break
  fi
done


