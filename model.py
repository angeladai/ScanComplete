"""Creates the scan completion model.

Takes as input a partial scan as a tsdf
in a dense volumetric grid and predicts a complete df (jointly with per-voxel
semantic labels, if specified). The model is hierarchical and autoregressive on
both previous hierarchy levels and within each hierarchy level prediction. Note
that the model is created for a single hierarchy level, and different models are
trained for each hierarchy level.
In particular, for a single hierarchy level, the model predicts a complete df
autoregressively by predicting eight interleaved voxel groups, each conditioned
on all previously predicted voxel groups. (Note that the eight voxel groups can
be trained in parallel with ground truth data.)
"""
# TODO add arxiv link when ready

import tensorflow as tf
import tensorflow.contrib.slim as slim

import constants
import util


def shortcut(inputs, num_input, num_output, stride):
  """Creates a shortcut (either a skip connection or a 1x1x1 convolution)."""
  if num_input == num_output:
    return inputs
  else:
    return slim.conv3d(
        inputs,
        num_outputs=num_output,
        kernel_size=[1, 1, 1],
        stride=[stride, stride, stride],
        activation_fn=None)


def model_block(inputs, num_channels_intermediate, num_channels_in, stride):
  """Creates a set of convs used as a block unit for the model."""
  b1 = slim.conv3d(
      inputs,
      num_outputs=num_channels_intermediate,
      kernel_size=[1, 1, 1],
      stride=[1, 1, 1],
      activation_fn=tf.nn.relu)
  b1 = slim.conv3d(
      b1,
      num_outputs=num_channels_intermediate,
      kernel_size=[3, 3, 3],
      stride=[stride, stride, stride],
      activation_fn=tf.nn.relu)
  b1 = slim.conv3d(
      b1,
      num_outputs=num_channels_intermediate * 4,
      kernel_size=[1, 1, 1],
      stride=[1, 1, 1],
      activation_fn=None)

  b2 = shortcut(inputs, num_channels_in, num_channels_intermediate * 4, stride)

  net = b1 + b2
  net = tf.nn.relu(net)
  return net


def model(input_scan,
          target_scan_low_resolution,
          target_scan,
          target_semantics_low_resolution,
          target_semantics,
          num_quant_levels,
          predict_semantics=False,
          use_p_norm=False):
  """Creates completion model.

  Args:
    input_scan: input partial scan as [abs(sdf),known(sdf)] in [-1,1]
    target_scan_low_resolution: previous low-resolution prediction (s)df as
                    [abs(sdf),known(sdf)] in [-1,1]; uses ground truth or
                    prediction(s) from previous level model at train time
    target_scan: previous voxel groups (s)df as [abs(sdf),known(sdf)] in [-1,1];
                 all eight groups are encoded as the full tensor; uses target at
                 train time
    target_semantics_low_resolution: previous low-resolution semantic class
                 predictions in [0, #classes); uses ground truth or
                 prediction(s) from previous model at train time
    target_semantics: previous semantic class voxel groups in [0, #classes); all
                eight groups are encoded as the full tensor; uses target at
                train time
    num_quant_levels: amount of quantization (if applicable)
    predict_semantics: whether to predict semantics
    use_p_norm: whether to use a deterministic prediction (p-norm loss)
  Returns:
    Returns a table of:
    logits_geo: list of predicted voxel groups (geometry / distance field)
    logits_sem: list of predicted voxel groups (semantic classes)
    features: features computed from input scan that are shared and can be
              cached (for other voxel groups at test time)
  """
  # Context from partial scan.
  context_scan = process_input(input_scan, 64, 64)
  context_scan = util.compute_voxel_groups(context_scan)

  # Context from previous hierarchy level.
  if target_scan_low_resolution is not None:
    context_previous_scan = process_input(target_scan_low_resolution, 32, 32)
  if predict_semantics and target_semantics_low_resolution is not None:
    context_previous_semantics = process_input(
        tf.cast(tf.expand_dims(target_semantics_low_resolution, 4), tf.float32)
        / constants.NUM_CLASSES * 2.0 - 1.0, 16, 32)

  if predict_semantics:
    groups_semantics = util.compute_voxel_groups(target_semantics)
    context_groups_semantics = process_previous_semantic_groups(
        groups_semantics,
        target_semantics.get_shape().as_list()[0], 32)

  # Context from previous groups.
  groups_geometry = util.compute_voxel_groups(target_scan)
  num_groups = len(groups_geometry)
  context_groups_geometry = process_previous_geo_groups(
      groups_geometry,
      target_scan.get_shape().as_list()[0], 64)

  predictions_geometry = []
  predictions_semantics = []
  cached_features = []
  # Model dependency among voxel groups.
  for n in range(num_groups):
    current_context_geometry = get_previous_voxel_group_features(
        context_groups_geometry, n)
    if predict_semantics:
      current_context_semantics = get_previous_voxel_group_features(
          context_groups_semantics, n)

    cached_features.append(context_scan[n])
    # Concatenate all features together and process.
    if n > 0:
      feature = tf.concat([context_scan[n], current_context_geometry], 4)
      if predict_semantics:
        feature = tf.concat([feature, current_context_semantics], 4)
    else:
      feature = context_scan[n]
    if target_scan_low_resolution is not None:
      feature = tf.concat([feature, context_previous_scan], 4)
    if predict_semantics and target_semantics_low_resolution is not None:
      feature = tf.concat([feature, context_previous_semantics], 4)

    num_channels = 64
    for _ in range(2):
      feature = model_block(feature, 32, num_channels, 1)
      num_channels *= 4

    # Split to semantics and geometry predictions.
    prediction_geometry = model_block(feature, 32, num_channels, 1)
    if predict_semantics:
      prediction_semantics = model_block(feature, 32, 32, 1)
      # Final convolution for semantics.
      prediction_semantics = slim.conv3d(
          prediction_semantics,
          num_outputs=constants.NUM_CLASSES,
          kernel_size=[1, 1, 1],
          stride=[1, 1, 1],
          activation_fn=None)
      predictions_semantics.append(prediction_semantics)
    # Final convolution for geometry.
    num_final_outputs = 1 if use_p_norm else num_quant_levels
    prediction_geometry = slim.conv3d(
        prediction_geometry,
        num_outputs=num_final_outputs,
        kernel_size=[1, 1, 1],
        stride=[1, 1, 1],
        activation_fn=None)
    predictions_geometry.append(prediction_geometry)

  # Return predicted voxel groups.
  return {
      'logits_geometry': predictions_geometry,
      'logits_semantics': predictions_semantics,
      'features': cached_features
  }


def process_input(inputs, num_channels_intermediate, num_channels_out):
  """Processes input tensors to model with some convs."""
  net = slim.conv3d(
      inputs,
      num_outputs=num_channels_intermediate,
      kernel_size=[3, 3, 3],
      stride=[1, 1, 1],
      activation_fn=tf.nn.relu)
  net = model_block(net, num_channels_intermediate, num_channels_out, 1)
  return net


def process_previous_semantic_groups(groups, batch_size, num_channels):
  """Processes previous voxel groups from semantic tensor."""
  num_groups = len(groups)
  groups = [tf.expand_dims(tf.expand_dims(x, 4), 1) for x in groups]
  groups = tf.concat(groups, 1)
  context_groups = tf.reshape(groups, [-1] + groups.get_shape().as_list()[2:])
  context_groups = (
      tf.cast(context_groups, tf.float32) / constants.NUM_CLASSES) * 2.0 - 1.0
  context_groups = slim.conv3d(
      context_groups,
      num_outputs=num_channels,
      kernel_size=[3, 3, 3],
      stride=[1, 1, 1],
      activation_fn=tf.nn.relu)
  context_groups = model_block(context_groups, num_groups, num_channels, 1)
  context_groups = tf.reshape(
      context_groups,
      [batch_size, num_groups] + context_groups.get_shape().as_list()[1:])
  return context_groups


def process_previous_geo_groups(groups, batch_size, num_channels):
  """Processes previous voxel groups from scan/geometry tensor."""
  num_groups = len(groups)
  groups = [tf.expand_dims(x, 1) for x in groups]
  groups = tf.concat(groups, 1)
  context_groups = tf.reshape(groups, [-1] + groups.get_shape().as_list()[2:])
  context_groups = slim.conv3d(
      context_groups,
      num_outputs=num_channels,
      kernel_size=[3, 3, 3],
      stride=[1, 1, 1],
      activation_fn=tf.nn.relu)
  context_groups = model_block(context_groups, num_groups, num_channels, 1)
  context_groups = tf.reshape(
      context_groups,
      [batch_size, num_groups] + context_groups.get_shape().as_list()[1:])
  return context_groups


def get_previous_voxel_group_features(context_groups, current_voxel_group):
  """Extracts prev voxel group features for current voxel group."""
  current_context = context_groups[:, :current_voxel_group, :, :, :, :]
  # Move previous voxel groups to the end and concat all together as features.
  current_context = tf.transpose(current_context, [0, 2, 3, 4, 1, 5])
  current_context = tf.reshape(current_context,
                               current_context.get_shape().as_list()[:-2] + [-1])
  return current_context