"""Util functions (preprocessing, quantization, save meshes, etc)."""

import os
import numpy as np
import scipy.io as sio
import tensorflow as tf
import constants

# Sdf/df preprocessing utility functions.


def quantize(tensor, num_quant_levels, truncation):
  """Quantizes df in tensor to [0,num_quant_levels-1]."""
  if num_quant_levels == 2:
    # Special case for occupancy grid (occupied voxels have df <= 1 voxel).
    return np.less_equal(tensor, 1).astype(np.uint8)
  return np.round((tensor / truncation) * (num_quant_levels - 1)).astype(
      np.uint8)


def dequantize(tensor, num_quant_levels, truncation):
  """De-quantizes tensor of [0,num_quant_levels-1] back to [0,truncation]."""
  if num_quant_levels == 2:
    # Convert to occupancy grid (occupied -> 0, empty -> 2).
    return np.not_equal(tensor, 1).astype(np.float32) * 2.0
  return tensor.astype(np.float32) * truncation / float(num_quant_levels - 1)


def preprocess_sdf(sdf, truncation):
  """Preprocesses sdf to [abs(tsdf),known(tsdf)] and put in [-1,1] for model."""
  sdf_abs = np.clip(np.abs(sdf), 0, truncation)
  sdf_abs = sdf_abs / (
      truncation / 2.0) - 1.0  # Put voxel context in range [-1,1].
  sdf_known = np.greater(sdf, -1).astype(float) * 2.0 - 1.0
  return np.stack([sdf_abs, sdf_known], 3)


def preprocess_df(df, truncation):
  """Preprocesses df by truncating to [0,truncation]."""
  return np.clip(df, 0, truncation)


def preprocess_target_sdf(sdf, num_quant_levels, truncation, apply_quantize):
  """Preprocesses target df/sdf to [abs(sdf),known(sdf)] in [-1,1] for model."""
  mask_known = np.greater_equal(sdf, -1)
  sdf = np.clip(np.abs(sdf), 0, truncation)
  if apply_quantize:
    sdf = quantize(sdf, num_quant_levels, truncation)
    sdf = sdf.astype(np.float32) / ((num_quant_levels - 1) / 2.0) - 1.0
  else:
    sdf = sdf / (constants.TRUNCATION / 2.0) - 1.0
  sdf_known = mask_known.astype(np.float32) * 2.0 - 1.0
  if len(sdf.shape) == 3:
    return np.stack([sdf, sdf_known], 3)
  return np.concatenate([sdf, sdf_known], 3)


def preprocess_target_sem(sem):
  """Preprocesses target sem (fix ceils labeled as floors)."""
  # Fixes wrong labels in the ground truth semantics.
  # Convert ceilings (2) labeled as floors to floor labels (4).
  # Detects wrong floor labels as those above half height.
  mid = sem.shape[1] // 2
  ceilings = np.ones(shape=sem[:, mid:, :].shape, dtype=np.uint8) * 2
  top = sem[:, mid:, :]
  bottom = sem[:, :mid, :]
  top = np.where(np.equal(top, 4), ceilings, top)
  return np.concatenate([bottom, top], 1)


# Visualization utility functions.


def make_label_color(label):
  """Provides default colors for semantics labels."""
  assert label >= 0 and label < constants.NUM_CLASSES
  return {
      0: [0.0, 0.0, 0.0],  # empty
      1: [240, 196, 135],  # bed
      2: [255, 160, 160],  # ceiling
      3: [214, 215, 111],  # chair
      4: [105, 170, 66],  # floor
      5: [229, 139, 43],  # furniture
      6: [201, 187, 223],  # objects
      7: [147, 113, 197],  # sofa
      8: [82, 131, 190],  # desk
      9: [172, 220, 31],  # tv
      10: [188, 228, 240],  # wall
      11: [140, 168, 215],  # window
      12: [128, 128, 128]  # unannotated
  }[int(label)]


def export_labeled_scene(pred_df, pred_sem, output_path, df_thresh=1):
  """Saves colored point cloud for semantics."""
  with open(output_path + '.obj', 'w') as output_file:
    for z in range(0, pred_df.shape[0]):
      for y in range(0, pred_df.shape[1]):
        for x in range(0, pred_df.shape[2]):
          if pred_df[z, y, x] > df_thresh:
            continue
          label = pred_sem[z, y, x]
          c = [ci / 255.0 for ci in make_label_color(label)]
          line = 'v %f %f %f %f %f %f\n' % (y, z, x, c[0], c[1], c[2])
          output_file.write(line)


def save_mat_df(df, error, filename):
  """Saves df as matlab .mat file."""
  output = {'x': df}
  if error is not None:
    output['errors'] = error
  sio.savemat(filename, output)


def save_iso_meshes(dfs, errs, semantics, filenames, isoval=1):
  """Saves dfs to obj files (by calling matlab's 'isosurface' function)."""
  assert len(dfs) == len(filenames) and (
      errs is None or len(dfs) == len(errs)) and (semantics is None or
                                                  len(dfs) == len(semantics))
  # Save semantics meshes if applicable.
  if semantics is not None:
    for i in range(len(filenames)):
      if semantics[i] is not None:
        export_labeled_scene(dfs[i], semantics[i],
                             os.path.splitext(filenames[i])[0] + '_sem')

  mat_filenames = [os.path.splitext(x)[0] + '.mat' for x in filenames]
  # Save .mat files for matlab call.
  command = ""
  for i in range(len(filenames)):
    if dfs[i] is None:
      continue
    err = None if errs is None else errs[i]
    save_mat_df(dfs[i], err, mat_filenames[i])
    command += "mat_to_obj('{0}', '{1}', {2});".format(mat_filenames[i],
                                                       filenames[i], isoval)
  command += 'exit;'

  tf.logging.info(
      'matlab -nodisplay -nosplash -nodesktop -r "{0}"'.format(command))
  # Execute matlab.
  os.system('matlab -nodisplay -nosplash -nodesktop -r "{0}"'.format(command))
  # Clean up .mat files.
  for i in range(len(mat_filenames)):
    os.system('rm -f {0}'.format(mat_filenames[i]))


# Export utility functions.


def float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, (tuple, list)):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, (tuple, list)):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  if not isinstance(value, (tuple, list)):
    value = [value]
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


# Voxel group utility functions.


def compute_voxel_group(tensor, group_id):
  """Extracts voxel group group_id (1-indexed) from (3, 4, or 5-dim) tensor."""
  assert group_id >= 1 and group_id <= 8
  group_id -= 1
  begin = [0, group_id / 4, group_id / 2 % 2, group_id % 2, 0]
  stride = [1, 2, 2, 2, 1]

  dim = len(tensor.shape)
  if dim == 3:
    begin = begin[1:4]
    stride = stride[1:4]
  elif dim == 4:
    begin = begin[:-1]
    stride = stride[:-1]

  return tf.strided_slice(tensor, begin, tensor.shape, stride)


def compute_voxel_groups(tensor):
  """Extracts list of all voxel groups from tensor."""
  groups = []
  for n in range(8):
    groups.append(compute_voxel_group(tensor, n + 1))
  return groups


def assign_voxel_group(dst, src, group_id):
  """Fills voxel group group_id of dst with src (uses channel 0 for ndim>3)."""
  assert group_id >= 1 and group_id <= 8
  group_id -= 1
  begin = [group_id / 4, group_id / 2 % 2, group_id % 2]
  dim = len(dst.shape)
  if dim == 3:
    dst[begin[0]::2, begin[1]::2, begin[2]::2] = src
  elif dim == 4:
    dst[0, begin[0]::2, begin[1]::2, begin[2]::2] = src
  elif dim == 5:
    dst[0, begin[0]::2, begin[1]::2, begin[2]::2, 0] = src
  else:
    raise
  return dst