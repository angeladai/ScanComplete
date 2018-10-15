#!/bin/bash

# Runs scan completion with hierarchical model over a set of scenes.

# Parameter section begins here. Edit to change number of test scenes, which model to use, output path.
MAX_NUM_TEST_SCENES=1000
NUM_HIERARCHY_LEVELS=3
BASE_OUTPUT_DIR=/home/github/ScanComplete/results_3frog

# Fill in path to test scenes
TEST_SCENES_PATH_3='/home/github/ScanComplete/data_3frog/vox19'
TEST_SCENES_PATH_2='/home/github/ScanComplete/data_3frog/vox9'
TEST_SCENES_PATH_1='/home/github/ScanComplete/data_3frog/vox5'

# Fill in model to use here
PREDICT_SEMANTICS=1
HIERARCHY_LEVEL_3_MODEL='/home/github/ScanComplete/models/completion-semantics/hierarchy3of3/'
HIERARCHY_LEVEL_2_MODEL='/home/github/ScanComplete/models/completion-semantics/hierarchy2of3/'
HIERARCHY_LEVEL_1_MODEL='/home/github/ScanComplete/models/completion-semantics/hierarchy1of3/'

# Specify output folders for each hierarchy level.
OUTPUT_FOLDER_3=${BASE_OUTPUT_DIR}/vis_level3/
OUTPUT_FOLDER_2=${BASE_OUTPUT_DIR}/vis_level2/
OUTPUT_FOLDER_1=${BASE_OUTPUT_DIR}/vis_level1/

# Specify padding (3frog specific)
# This has to be increased for every finer level
PADDING_LEVEL_3=2
PADDING_LEVEL_2=4
PADDING_LEVEL_1=8

# Specify maximum height of input
# This has to be increased for every finer level
HEIGHT_INPUT_LEVEL_3=32
HEIGHT_INPUT_LEVEL_2=64
HEIGHT_INPUT_LEVEL_1=128

# End parameter section.

# Run hierarchy.

# ------- hierarchy level 1 ------- #

IS_BASE_LEVEL=1
HIERARCHY_LEVEL=1

# Go through all test scenes.
count=1
REC_COUNTER=0
for scene in $TEST_SCENES_PATH_3/*.tfrecords; do
  ((REC_COUNTER++))
done
TOTAL_SCENES=$(($REC_COUNTER<$MAX_NUM_TEST_SCENES?$REC_COUNTER:$MAX_NUM_TEST_SCENES))
for scene in $TEST_SCENES_PATH_3/*.tfrecords; do
  echo "Processing hierarchy level 3, scene $count of $TOTAL_SCENES: $scene".
  python complete_scan_3frog.py \
    --alsologtostderr \
    --base_dir="${HIERARCHY_LEVEL_3_MODEL}" \
    --height_input="${HEIGHT_INPUT_LEVEL_3}" \
    --hierarchy_level="${HIERARCHY_LEVEL}" \
    --num_total_hierarchy_levels="${NUM_HIERARCHY_LEVELS}" \
    --is_base_level="${IS_BASE_LEVEL}" \
    --predict_semantics="${PREDICT_SEMANTICS}" \
    --output_folder="${OUTPUT_FOLDER_3}" \
    --input_scene="${scene}" \
    --pad_test="${PADDING_LEVEL_3}"
  ((count++))
  if (( count > MAX_NUM_TEST_SCENES )); then
    break
  fi
done

# ------- hierarchy level 2 ------- #

IS_BASE_LEVEL=0
HIERARCHY_LEVEL=2

# go thru all test scenes
count=1
REC_COUNTER=0
for scene in $TEST_SCENES_PATH_2/*.tfrecords; do
  ((REC_COUNTER++))
done
TOTAL_SCENES=$(($REC_COUNTER<$MAX_NUM_TEST_SCENES?$REC_COUNTER:$MAX_NUM_TEST_SCENES))
for scene in $TEST_SCENES_PATH_2/*.tfrecords; do
  echo "Processing hierarchy level 2, scene $count of $TOTAL_SCENES: $scene".
  python complete_scan_3frog.py \
    --alsologtostderr \
    --base_dir="${HIERARCHY_LEVEL_2_MODEL}" \
    --output_dir_prev="${OUTPUT_FOLDER_3}" \
    --height_input="${HEIGHT_INPUT_LEVEL_2}" \
    --hierarchy_level="${HIERARCHY_LEVEL}" \
    --num_total_hierarchy_levels="${NUM_HIERARCHY_LEVELS}" \
    --is_base_level="${IS_BASE_LEVEL}" \
    --predict_semantics="${PREDICT_SEMANTICS}" \
    --output_folder="${OUTPUT_FOLDER_2}" \
    --input_scene="${scene}" \
    --pad_test="${PADDING_LEVEL_2}"
  ((count++))
  if (( count > MAX_NUM_TEST_SCENES )); then
    break
  fi
done

# ------- hierarchy level 3 ------- #

IS_BASE_LEVEL=0
HIERARCHY_LEVEL=3

# go thru all test scenes
count=1
REC_COUNTER=0
for scene in $TEST_SCENES_PATH_1/*.tfrecords; do
  ((REC_COUNTER++))
done
TOTAL_SCENES=$(($REC_COUNTER<$MAX_NUM_TEST_SCENES?$REC_COUNTER:$MAX_NUM_TEST_SCENES))
for scene in $TEST_SCENES_PATH_1/*.tfrecords; do
  echo "Processing hierarchy level 1, scene $count of $TOTAL_SCENES: $scene".
  python complete_scan_3frog.py \
    --alsologtostderr \
    --base_dir="${HIERARCHY_LEVEL_1_MODEL}" \
    --output_dir_prev="${OUTPUT_FOLDER_2}" \
    --height_input="${HEIGHT_INPUT_LEVEL_1}" \
    --hierarchy_level="${HIERARCHY_LEVEL}" \
    --num_total_hierarchy_levels="${NUM_HIERARCHY_LEVELS}" \
    --is_base_level="${IS_BASE_LEVEL}" \
    --predict_semantics="${PREDICT_SEMANTICS}" \
    --output_folder="${OUTPUT_FOLDER_1}" \
    --input_scene="${scene}" \
    --pad_test="${PADDING_LEVEL_1}"
  ((count++))
  if (( count > MAX_NUM_TEST_SCENES )); then
    break
  fi
done


