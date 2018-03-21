"""Losses."""

import tensorflow as tf
import tensorflow.contrib.slim as slim

import constants


def get_l1_loss_allgroups(logit_groups,
                          labels,
                          logit_groups_sem,
                          labels_sem,
                          weight_semantic=1):
  """Builds (deterministic) l1 reconstruction loss + (optional) semantic loss.

  Args:
    logit_groups: A list of tensors representing the predicted voxel groups of a
                  scan.
    labels: A tensor representing the ground truth scan.
    logit_groups_sem: A list of tensors representing the predicted voxel group
                      semantics of a scan.
    labels_sem: A tensor representing the ground truth semantics of a scan.
    weight_semantic: Amount to weight semantic loss relative to geometric.
  Returns:
    The loss function as Tensorflow op.
  """
  mask = tf.cast(tf.equal(labels[:, :, :, :, 1], 1), tf.float32)
  labels = labels[:, :, :, :, 0]
  label_groups = [
      labels[:, ::2, ::2, ::2],
      labels[:, ::2, ::2, 1::2],
      labels[:, ::2, 1::2, ::2],
      labels[:, ::2, 1::2, 1::2],
      labels[:, 1::2, ::2, ::2],
      labels[:, 1::2, ::2, 1::2],
      labels[:, 1::2, 1::2, ::2],
      labels[:, 1::2, 1::2, 1::2],
  ]
  weight_groups = [
      mask[:, ::2, ::2, ::2],
      mask[:, ::2, ::2, 1::2],
      mask[:, ::2, 1::2, ::2],
      mask[:, ::2, 1::2, 1::2],
      mask[:, 1::2, ::2, ::2],
      mask[:, 1::2, ::2, 1::2],
      mask[:, 1::2, 1::2, ::2],
      mask[:, 1::2, 1::2, 1::2],
  ]

  loss = 0.0
  for n in range(len(label_groups)):
    loss += tf.losses.absolute_difference(
        label_groups[n],
        logit_groups[n][:, :, :, :, 0],
        weights=weight_groups[n])
  loss /= len(label_groups)
  loss_geo = loss

  loss_sem = 0.0
  if logit_groups_sem and weight_semantic > 0:
    loss_sem = get_loss_semantic_allgroups(logit_groups_sem, labels_sem,
                                           weight_groups)
    loss += weight_semantic * loss_sem  # scale to similar
  return {'loss': loss, 'loss_geo': loss_geo, 'loss_sem': loss_sem}


def get_loss_semantic_allgroups(logit_groups_sem, labels_sem, weight_groups):
  """Builds semantic softmax cross-entropy loss.

  Args:
    logit_groups_sem: A list of tensors representing the predicted voxel group
                      semantics of a scan.
    labels_sem: A tensor representing the ground truth semantics of a scan.
    weight_groups: A list of tensors weighting the loss for each group.
  Returns:
    The loss function as Tensorflow op.
  """
  one_hot_labels = slim.one_hot_encoding(
      labels_sem, constants.NUM_CLASSES, on_value=1.0, off_value=0.0)
  labels = [
      one_hot_labels[:, ::2, ::2, ::2],
      one_hot_labels[:, ::2, ::2, 1::2],
      one_hot_labels[:, ::2, 1::2, ::2],
      one_hot_labels[:, ::2, 1::2, 1::2],
      one_hot_labels[:, 1::2, ::2, ::2],
      one_hot_labels[:, 1::2, ::2, 1::2],
      one_hot_labels[:, 1::2, 1::2, ::2],
      one_hot_labels[:, 1::2, 1::2, 1::2],
  ]
  for n in range(len(labels)):
    weight_groups[n] = tf.multiply(
        weight_groups[n],
        tf.reduce_sum(tf.multiply(labels[n], constants.WEIGHT_CLASSES), 4))
  loss = 0.0
  for n in range(len(labels)):
    loss += tf.losses.softmax_cross_entropy(
        onehot_labels=labels[n],
        logits=logit_groups_sem[n],
        weights=weight_groups[n])
  loss /= len(labels)
  return loss


def get_probabilistic_loss_allgroups(logit_groups,
                                     labels,
                                     logit_groups_sem,
                                     labels_sem,
                                     num_quant_levels,
                                     weight_semantic=1):
  """Builds cross-entropy reconstruction loss + (optional) semantic loss.

  Args:
    logit_groups: A list of tensors representing the predicted voxel groups of a
                  scan.
    labels: A tensor representing the ground truth scan.
    logit_groups_sem: A list of tensors representing the predicted voxel group
                      semantics of a scan.
    labels_sem: A tensor representing the ground truth semantics of a scan.
    num_quant_levels: Number of quantization bins.
    weight_semantic: Amount to weight semantic loss relative to geometric.
  Returns:
    The loss function as Tensorflow op.
  """
  mask = tf.cast(tf.equal(labels[:, :, :, :, 1], 1), tf.float32)
  labels = tf.cast((labels[:, :, :, :, 0] + 1) * 0.5 * (num_quant_levels - 1),
                   tf.uint8)

  one_hot_labels = slim.one_hot_encoding(
      labels, num_quant_levels, on_value=1.0, off_value=0.0)

  labels = [
      one_hot_labels[:, ::2, ::2, ::2, :],
      one_hot_labels[:, ::2, ::2, 1::2, :],
      one_hot_labels[:, ::2, 1::2, ::2, :],
      one_hot_labels[:, ::2, 1::2, 1::2, :],
      one_hot_labels[:, 1::2, ::2, ::2, :],
      one_hot_labels[:, 1::2, ::2, 1::2, :],
      one_hot_labels[:, 1::2, 1::2, ::2, :],
      one_hot_labels[:, 1::2, 1::2, 1::2, :],
  ]
  weight_groups = [
      mask[:, ::2, ::2, ::2],
      mask[:, ::2, ::2, 1::2],
      mask[:, ::2, 1::2, ::2],
      mask[:, ::2, 1::2, 1::2],
      mask[:, 1::2, ::2, ::2],
      mask[:, 1::2, ::2, 1::2],
      mask[:, 1::2, 1::2, ::2],
      mask[:, 1::2, 1::2, 1::2],
  ]

  loss = 0.0
  for n in range(len(labels)):
    loss += tf.losses.softmax_cross_entropy(
        onehot_labels=labels[n],
        logits=logit_groups[n],
        weights=weight_groups[n])
  loss /= len(labels)
  loss_geo = loss

  loss_sem = 0.0
  if logit_groups_sem and weight_semantic > 0:
    loss_sem = get_loss_semantic_allgroups(logit_groups_sem, labels_sem,
                                           weight_groups)
    loss += weight_semantic * loss_sem

  return {'loss': loss, 'loss_geo': loss_geo, 'loss_sem': loss_sem}


def get_recon_loss(pred, target):
  """Computes reconstruction error (l1) over entire grid."""
  return tf.losses.absolute_difference(labels=target, predictions=pred)


def get_recon_loss_for_occupied_space(tensor, reference, df_thresh):
  """Computes reconstruction error (l1) over occupied space of reference."""
  # Use df_thresh as threshold for determining occupied space
  mask = tf.less(reference, df_thresh)
  return tf.losses.absolute_difference(
      labels=tf.boolean_mask(tensor=reference, mask=mask),
      predictions=tf.boolean_mask(tensor=tensor, mask=mask))


def get_recon_loss_for_known(pred, target, input_sdf):
  """Computes reconstruction error (l1) over known space of input scan."""
  # input_sdf is the processed [abs,known] representation.
  mask = tf.equal(input_sdf[:, :, :, 1], 1)
  return tf.losses.absolute_difference(
      labels=tf.boolean_mask(tensor=pred, mask=mask),
      predictions=tf.boolean_mask(tensor=target, mask=mask))


def get_recon_loss_for_unknown(pred, target, input_sdf):
  """Computes reconstruction error (l1) over unknown space of input scan."""
  # input_sdf is the processed [abs,known] representation.
  mask = tf.equal(input_sdf[:, :, :, 1], -1)
  return tf.losses.absolute_difference(
      labels=tf.boolean_mask(tensor=pred, mask=mask),
      predictions=tf.boolean_mask(tensor=target, mask=mask))