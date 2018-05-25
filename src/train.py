"""Builds model and trains it."""

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import constants
import losses
import model
import preprocessor
import reader

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('gpu', 3, 'GPU to run on')
flags.DEFINE_integer('batch_size', 8, 'Number of blocks in each batch.')
flags.DEFINE_string('data_filepattern', '/tmp/data/train_*.tfrecords',
                    'Training data file pattern.')
flags.DEFINE_integer('dim_block', 32, 'Input/target block x/z dim.')
flags.DEFINE_integer('height_block', 16, 'Input/target block y dim.')
flags.DEFINE_integer('hierarchy_level', 1, 'Hierachy level (1: finest level)')
flags.DEFINE_bool('is_base_level', True,
                  'Whether there is a previous hierarchy level.')
flags.DEFINE_integer('p_norm', 1, 'p-norm loss (0 to disable).')
flags.DEFINE_bool('predict_semantics', False,
                  'Also predict semantic labels per-voxel.')
flags.DEFINE_integer('num_quant_levels', 256, 'Number of quantization bins.')
flags.DEFINE_integer('stored_dim_block', 64,
                     'Stored data block x/z dim, high-resolution.')
flags.DEFINE_integer('stored_height_block', 64,
                     'Stored data block y dim, high-resolution.')
flags.DEFINE_float('weight_semantic', 0.5, 'Weight for semantic loss.')

# Train optimization parameters.
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
flags.DEFINE_float('min_learning_rate', 0.0001, 'Min learning rate.')
flags.DEFINE_integer('number_of_steps', 100000, '#train iters.')
flags.DEFINE_string('train_dir', '/tmp/scene_complete/train',
                    'The root dir of train.')
flags.DEFINE_bool('train_samples', False, 'Train on previous model prediction.')

os.environ['CUDA_VISIBLE_DEVICES']=str(FLAGS.gpu)
tf.logging.set_verbosity(tf.logging.INFO)
analyzer = tf.contrib.tfprof.model_analyzer


def _train():
  """Training loop."""
  params = dict()
  params['batch_size'] = FLAGS.batch_size
  params['learning_rate'] = FLAGS.learning_rate

  with tf.device('/gpu:0'):
    global_step = slim.create_global_step()

  with tf.device('/gpu:0'):
    lrn_rate = tf.maximum(
        FLAGS.min_learning_rate,  # min_lr_rate.
        tf.train.exponential_decay(
            params['learning_rate'], global_step, 5000, 0.92, staircase=True))
    tf.summary.scalar('learning_rate', lrn_rate)
    optimizer = tf.train.AdamOptimizer(lrn_rate)

  with tf.device('/cpu:0'):
    (input_blocks, target_blocks, target_sem_blocks, target_lo_blocks,
     target_sem_lo_blocks) = reader.ReadSceneBlocksLevel(
         FLAGS.data_filepattern,
         FLAGS.train_samples,
         FLAGS.dim_block,
         FLAGS.height_block,
         FLAGS.stored_dim_block,
         FLAGS.stored_height_block,
         FLAGS.is_base_level,
         FLAGS.hierarchy_level,
         FLAGS.num_quant_levels,
         quantize=not FLAGS.p_norm,
         params=params,
         shuffle=True)

    if FLAGS.is_base_level:
      inputs_queue = slim.python.slim.data.prefetch_queue.prefetch_queue(
          (input_blocks, target_blocks, target_sem_blocks))
    else:
      inputs_queue = slim.python.slim.data.prefetch_queue.prefetch_queue(
          (input_blocks, target_blocks, target_sem_blocks, target_lo_blocks,
           target_sem_lo_blocks))

  def tower_fn(inputs_queue):
    """The tower function."""
    target_lo_blocks = None
    target_sem_blocks = None
    target_sem_lo_blocks = None
    if FLAGS.is_base_level:
      input_blocks, target_blocks, target_sem_blocks = inputs_queue.dequeue()
    else:
      (input_blocks, target_blocks, target_sem_blocks, target_lo_blocks,
       target_sem_lo_blocks) = inputs_queue.dequeue()
      
    ops = model.model(
        input_scan=input_blocks,
        target_scan_low_resolution=target_lo_blocks,
        target_scan=target_blocks,
        target_semantics_low_resolution=target_sem_lo_blocks,
        target_semantics=target_sem_blocks,
        predict_semantics=FLAGS.predict_semantics,
        use_p_norm=FLAGS.p_norm > 0,
        num_quant_levels=FLAGS.num_quant_levels)
    logits = ops['logits_geometry']
    logits_sem = ops['logits_semantics']

    # TODO(angeladai) change p-norm to l1
    if FLAGS.p_norm > 0:
      loss = losses.get_l1_loss_allgroups(
          logit_groups=logits,
          labels=target_blocks,
          logit_groups_sem=logits_sem,
          labels_sem=target_sem_blocks,
          weight_semantic=FLAGS.weight_semantic)
    else:
      loss = losses.get_probabilistic_loss_allgroups(
          logit_groups=logits,
          labels=target_blocks,
          logit_groups_sem=logits_sem,
          labels_sem=target_sem_blocks,
          num_quant_levels=FLAGS.num_quant_levels,
          weight_semantic=FLAGS.weight_semantic)
    if FLAGS.predict_semantics:
      tf.summary.scalar('Loss_Geo', loss['loss_geo'])
      tf.summary.scalar('Loss_Sem', loss['loss_sem'])

    # Reconstruct
    predictions_list = []
    temp = 100.0
    for l in logits:
      if FLAGS.p_norm > 0:
        predictions_list.append(l[:, :, :, :, 0])
      else:
        sz = l.shape_as_list()
        l = tf.reshape(l, [-1, sz[-1]])
        s = tf.multinomial(temp * l, 1)
        predictions_list.append(tf.reshape(s, sz[:-1]))
    if FLAGS.predict_semantics:
      target_sem_groups = [
          target_sem_blocks[:, ::2, ::2, ::2],
          target_sem_blocks[:, ::2, ::2, 1::2],
          target_sem_blocks[:, ::2, 1::2, ::2],
          target_sem_blocks[:, ::2, 1::2, 1::2],
          target_sem_blocks[:, 1::2, ::2, ::2],
          target_sem_blocks[:, 1::2, ::2, 1::2],
          target_sem_blocks[:, 1::2, 1::2, ::2],
          target_sem_blocks[:, 1::2, 1::2, 1::2]
      ]
      error_count = error_count_1 = 0
      error_norm = 0
      for n in range(len(logits_sem)):
        pred_sem = tf.argmax(logits_sem[n], 4)
        mask = tf.greater(target_sem_groups[n], 0)
        error_count += tf.count_nonzero(
            tf.cast(tf.boolean_mask(tensor=pred_sem, mask=mask), tf.int32) -
            tf.cast(
                tf.boolean_mask(tensor=target_sem_groups[n], mask=mask),
                tf.int32))
        error_norm += tf.count_nonzero(mask)
        if n == 0:
          error_count_1 = tf.cast(error_count, tf.float32) / tf.cast(
              error_norm, tf.float32)
      tf.summary.scalar(
          'Sem_Accuracy', 1.0 -
          tf.cast(error_count, tf.float32) / tf.cast(error_norm, tf.float32))
      tf.summary.scalar('Sem_Accuracy_Group_1', 1.0 - error_count_1)

    target_groups = [
        target_blocks[:, ::2, ::2, ::2, 0], target_blocks[:, ::2, ::2, 1::2, 0],
        target_blocks[:, ::2, 1::2, ::2, 0],
        target_blocks[:, ::2, 1::2, 1::2, 0],
        target_blocks[:, 1::2, ::2, ::2, 0],
        target_blocks[:, 1::2, ::2, 1::2, 0],
        target_blocks[:, 1::2, 1::2, ::2, 0],
        target_blocks[:, 1::2, 1::2, 1::2, 0]
    ]
    l1_recon_loss = 0.0
    recon_using_pred_occ = 0.0
    recon_using_target_occ = 0.0
    for k in range(len(predictions_list)):
      if FLAGS.p_norm > 0:
        p = (predictions_list[k] + 1) * 0.5 * constants.TRUNCATION
        t = (target_groups[k] + 1) * 0.5 * constants.TRUNCATION
      else:
        p = preprocessor.dequantize(predictions_list[k], FLAGS.num_quant_levels,
                                    constants.TRUNCATION)
        t = preprocessor.dequantize(
            (target_groups[k] + 1) * 0.5 * FLAGS.num_quant_levels,
            FLAGS.num_quant_levels, constants.TRUNCATION)
      l1_recon_loss += losses.get_recon_loss(pred=p, target=t)
      # for occupied space error use 1.5 to include transition from occ to empty
      recon_using_pred_occ += losses.get_recon_loss_for_occupied_space(
          t, p, 1.5)
      recon_using_target_occ += losses.get_recon_loss_for_occupied_space(
          p, t, 1.5)
    l1_recon_loss /= len(predictions_list)
    recon_using_pred_occ /= len(predictions_list)
    recon_using_target_occ /= len(predictions_list)
    tf.summary.scalar('Recon_Loss', l1_recon_loss)
    tf.summary.scalar('Recon_Loss_From_Pred_Occ', recon_using_pred_occ)
    tf.summary.scalar('Recon_Loss_From_Target_Occ', recon_using_target_occ)

    return {'logits': logits, 'loss': loss['loss']}

  with tf.device('/gpu:0'):
    total_loss = tower_fn(inputs_queue)['loss']
    tf.summary.scalar('Total_Loss', total_loss)

  session_config = tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=False)
  train_op = slim.learning.create_train_op(total_loss, optimizer)

  # Run training.
  tf.logging.info('Running training')
  slim.learning.train(
      train_op,
      FLAGS.train_dir,
      session_config=session_config,
      save_summaries_secs=60,
      save_interval_secs=180,
      number_of_steps=FLAGS.number_of_steps,
      saver=tf.train.Saver(keep_checkpoint_every_n_hours=1., max_to_keep=50))


def main(_):
  _train()


if __name__ == '__main__':
  tf.app.run()
