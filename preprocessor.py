"""Util functions (preprocessing, quantization, save meshes, etc)."""

# TODO merge with util

import tensorflow as tf

import constants

# Train block preprocessing.


def extract_block(tensor,
                  block_dim_x,
                  block_dim_y,
                  block_dim_z,
                  subsample_factor,
                  height_jitter=0):
  """Extract train block from stored data block."""
  sz = tensor.get_shape().as_list()
  assert len(sz) == 4 or len(sz) == 3
  sz = sz[0:3]
  block_dim_in_tensor_x = block_dim_x * subsample_factor
  block_dim_in_tensor_y = block_dim_y * subsample_factor
  block_dim_in_tensor_z = block_dim_z * subsample_factor
  # y starts from zero since it is the height axis, don't crop there.
  offset_x = (sz[0] - block_dim_in_tensor_x) / 2
  offset_y = height_jitter
  offset_z = (sz[2] - block_dim_in_tensor_z) / 2
  block = tensor[offset_x:offset_x + block_dim_in_tensor_x:subsample_factor,
                 offset_y:offset_y + block_dim_in_tensor_y:subsample_factor,
                 offset_z:offset_z + block_dim_in_tensor_z:subsample_factor]
  return block


def quantize(tensor, num_quant_levels, truncation):
  """Quantizes df in tensor to [0,num_quant_levels-1]."""
  if num_quant_levels == 2:
    # Special case for occupancy grid (occupied voxels have df <= 1 voxel).
    return tf.cast(tf.less_equal(tensor, 1), tf.uint8)
  return tf.cast(
      tf.round((tensor / truncation) * (num_quant_levels - 1)), tf.uint8)


def dequantize(tensor, num_quant_levels, truncation):
  """De-quantizes tensor of [0,num_quant_levels-1] back to [0,truncation]."""
  if num_quant_levels == 2:
    # Convert to occupancy grid (occupied -> 0, empty -> 2).
    return tf.cast(tf.not_equal(tensor, 1), tf.float32) * 2.0
  return tf.cast(tensor, tf.float32) * truncation / float(num_quant_levels - 1)


def preprocess_sdf(sdf, truncation):
  """Preprocesses sdf to [abs(tsdf),known(tsdf)] and put in [-1,1] for model."""
  sdf_abs = tf.clip_by_value(tf.abs(tf.cast(sdf, tf.float32)), 0, truncation)
  sdf_abs = sdf_abs / (
      truncation / 2.0) - 1.0  # Put voxel context in range [-1,1].
  sdf_known = tf.cast(tf.greater(sdf, -1), tf.float32) * 2.0 - 1.0
  return tf.concat([sdf_abs, sdf_known], 3)


def preprocess_target_sdf(sdf, num_quant_levels, truncation, apply_quantize):
  """Preprocesses target df/sdf to [abs(sdf),known(sdf)] in [-1,1] for model."""
  mask_known = tf.greater_equal(sdf, -1)
  sdf = tf.clip_by_value(tf.abs(tf.cast(sdf, tf.float32)), 0, truncation)
  if apply_quantize:
    sdf = quantize(sdf, num_quant_levels, truncation)
    sdf = tf.cast(sdf, tf.float32) / ((num_quant_levels - 1) / 2.0) - 1.0
  else:
    sdf = sdf / (constants.TRUNCATION / 2.0) - 1.0
  sdf_known = tf.cast(mask_known, tf.float32) * 2.0 - 1.0
  if len(sdf.shape) == 3:
    return tf.stack([sdf, sdf_known], 3)
  return tf.concat([sdf, sdf_known], 3)


def preprocess_target_sem(sem):
  """Preprocesses target sem (fix ceils labeled as floors)."""
  # Fixes wrong labels in the ground truth semantics.
  # Convert ceilings (2) labeled as floors to floor labels (4).
  # Detects wrong floor labels as those above half height.
  mid = sem.shape[1] // 2
  ceilings = tf.ones(shape=sem[:, mid:, :].shape, dtype=tf.uint8) * 2
  top = sem[:, mid:, :]
  bottom = sem[:, :mid, :]
  top = tf.where(tf.equal(top, 4), ceilings, top)
  return tf.concat([bottom, top], 1)