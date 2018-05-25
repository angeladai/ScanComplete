"""Reader for generating train batches from TFRecord files."""

import numpy as np
import tensorflow as tf

import constants
import preprocessor

_RESOLUTIONS = ['5cm', '9cm', '19cm']
_INPUT_FEATURE = 'input_sdf'
_TARGET_FEATURE = 'target_df'
_TARGET_SEM_FEATURE = 'target_sem'
_HEIGHT_JITTER = [5, 3, 0]


def ReadSceneBlocksLevel(data_filepattern,
                         train_samples,
                         dim_block,
                         height_block,
                         stored_dim_block,
                         stored_height_block,
                         is_base_level,
                         hierarchy_level,
                         num_quant_levels,
                         params,
                         quantize=False,
                         shuffle=True,
                         num_epochs=None):
  """Reads a batch of block examples from the SSTable files.

  Args:
    data_filepattern: String matching input files.
    train_samples: Train on previous model predictions.
    dim_block: x/z dimension of train block.
    height_block: y dimension of train block.
    stored_dim_block: Stored data x/z dimension (high-resolution).
    stored_height_block: Stored data y dimension (high-resolution).
    is_base_level: Whether there are no previous hierarchy levels.
    hierarchy_level: hierarchy level (1 is finest).
    num_quant_levels: Number of quantization bins (if used).
    params: Parameter dictionary.
    quantize: Whether to quantize data.
    shuffle: Whether to shuffle.
    num_epochs: Number of times to go over the input.
  Returns:
    5-D tensor input batches of shape
      [batch_size, dim_input, height_input, dim_input, 2], dtype
      tf.float32.
    5-D tensor target batches of shape
      [batch_size, dim_target, height_target, dim_target, 2], dtype
      tf.float32.
    5-D tensor target semantics batches of shape
      [batch_size, dim_target, height_target, dim_target], dtype tf.uint8.
    (optionally) 5-D tensor low-resolution target batches of shape
      [batch_size, dim_target//2, height_target//2, dim_target//2, 2], dtype
      tf.float32.
    (optionally) 5-D tensor low-resolution target semantics batches of shape
      [batch_size, dim_target//2, height_target//2, dim_target//2], dtype
      tf.uint8.
  """
  assert (stored_dim_block >= dim_block and
          stored_height_block >= height_block)

  read_target_lo = not is_base_level
  _, examples, samples_lo, samples_sem_lo = _ReadBlockExample(
      data_filepattern,
      train_samples=train_samples,
      hierarchy_level=hierarchy_level,
      is_base_level=is_base_level,
      shuffle=shuffle,
      num_epochs=num_epochs,
      stored_dim_block=stored_dim_block,
      stored_height_block=stored_height_block)

  # jitter height (must be even)
  jitter = (np.random.random_integers(
      low=0, high=_HEIGHT_JITTER[hierarchy_level - 1]) // 2) * 2

  # extract relevant portion of data block as per input/target dim
  key_input = _RESOLUTIONS[hierarchy_level - 1] + '_' + _INPUT_FEATURE
  key_target = _RESOLUTIONS[hierarchy_level - 1] + '_' + _TARGET_FEATURE
  key_target_sem = _RESOLUTIONS[hierarchy_level - 1] + '_' + _TARGET_SEM_FEATURE
  #key_input = 'input_sdf'
  #key_target = 'target_df'
  #key_target_sem = 'target_sem'

  input_sdf_blocks = examples[key_input]
  input_sdf_blocks = preprocessor.extract_block(
      input_sdf_blocks, dim_block, height_block, dim_block, 1, jitter)
  input_blocks = preprocessor.preprocess_sdf(input_sdf_blocks,
                                             constants.TRUNCATION)

  target_df_blocks = examples[key_target]
  target_df_blocks = preprocessor.extract_block(
      target_df_blocks, dim_block, height_block, dim_block, 1, jitter)
  target_blocks = preprocessor.preprocess_target_sdf(
      target_df_blocks, num_quant_levels, constants.TRUNCATION, quantize)

  if read_target_lo:
    if train_samples:
      target_lo_blocks = samples_lo
    else:
      key_target_lo = _RESOLUTIONS[hierarchy_level] + '_' + _TARGET_FEATURE
      target_lo_blocks = examples[key_target_lo]
    target_lo_blocks = preprocessor.extract_block(
        target_lo_blocks, dim_block // 2, height_block // 2, dim_block // 2, 1,
        jitter // 2)
    target_lo_blocks = preprocessor.preprocess_target_sdf(
        target_lo_blocks, num_quant_levels, constants.TRUNCATION, quantize)

  target_sem_blocks = tf.decode_raw(examples[key_target_sem], tf.uint8)
  target_sem_blocks = tf.reshape(
      target_sem_blocks,
      [stored_dim_block, stored_height_block, stored_dim_block])
  target_sem_blocks = preprocessor.extract_block(
      target_sem_blocks, dim_block, height_block, dim_block, 1, jitter)
  target_sem_blocks = preprocessor.preprocess_target_sem(target_sem_blocks)
  if read_target_lo:
    if train_samples:
      target_sem_lo_blocks = tf.decode_raw(samples_sem_lo, tf.uint8)
    else:
      key_target_sem_lo = (
          _RESOLUTIONS[hierarchy_level] + '_' + _TARGET_SEM_FEATURE)
      target_sem_lo_blocks = tf.decode_raw(examples[key_target_sem_lo],
                                           tf.uint8)
    target_sem_lo_blocks = tf.reshape(target_sem_lo_blocks, [
        stored_dim_block // 2, stored_height_block // 2, stored_dim_block // 2
    ])
    target_sem_lo_blocks = preprocessor.extract_block(
        target_sem_lo_blocks, dim_block // 2, height_block // 2, dim_block // 2,
        1, jitter // 2)
    if not train_samples:
      target_sem_lo_blocks = preprocessor.preprocess_target_sem(
          target_sem_lo_blocks)

  blocks = [input_blocks, target_blocks, target_sem_blocks]
  if read_target_lo:
    blocks.append(target_lo_blocks)
    blocks.append(target_sem_lo_blocks)

  if shuffle:
    batched = tf.train.shuffle_batch(
        blocks,
        batch_size=params['batch_size'],
        num_threads=64,
        capacity=params['batch_size'] * 100,
        min_after_dequeue=params['batch_size'] * 4)
  else:
    batched = tf.train.batch(
        blocks,
        batch_size=params['batch_size'],
        num_threads=16,
        capacity=params['batch_size'])

  input_blocks = batched[0]
  target_blocks = batched[1]
  target_sem_blocks = batched[2]
  target_lo_blocks = None
  target_sem_lo_blocks = None
  if read_target_lo:
    target_lo_blocks = batched[3]
    target_sem_lo_blocks = batched[4]
  return (input_blocks, target_blocks, target_sem_blocks, target_lo_blocks,
          target_sem_lo_blocks)


def _ReadBlockExample(data_filepattern, train_samples, hierarchy_level,
                      is_base_level, stored_dim_block,
                      stored_height_block, shuffle, num_epochs):
  """Deserializes train data.

  Args:
    data_filepattern: A list of data file patterns.
    train_samples: Train on previous model predictions.
    hierarchy_level: hierarchy level (1 is finest).
    is_base_level: Whether there are no previous hierarchy levels.
    stored_dim_block: Stored data x/z dimension (high-resolution).
    stored_height_block: Stored data y dimension (high-resolution).
    shuffle: Whether to shuffle.
    num_epochs: Number of data epochs.
  Returns:
    key, value.
  """

  if not isinstance(data_filepattern, list):
    data_filepattern = [data_filepattern]
  # Get filenames matching filespec.
  tf.logging.info('data_filepattern: %s', data_filepattern)
  filenames = []
  for p in data_filepattern:
    filenames.extend(tf.gfile.Glob(p))
  tf.logging.info('filenames: %s', filenames)

  # Create filename queue.
  filename_queue = tf.train.string_input_producer(
      filenames, num_epochs=num_epochs, shuffle=shuffle)

  # Read sequence examples.
  reader = tf.TFRecordReader()
  key, serialized_example = reader.read(filename_queue)

  samples_lo = None
  samples_sem_lo = None
  if train_samples and not is_base_level:
    key_samples = 'samples_' + _RESOLUTIONS[hierarchy_level] 
    key_samples_sem = 'sem_samples_' + _RESOLUTIONS[hierarchy_level] 
    spec = {
        'data':
            tf.FixedLenFeature((), tf.string),
        key_samples:
            tf.FixedLenFeature((stored_dim_block // 2, stored_height_block // 2,
                                stored_dim_block // 2, 1), tf.float32),
        key_samples_sem:
            tf.FixedLenFeature((), tf.string)
    }
    example = tf.parse_single_example(serialized_example, spec)
    samples_lo = example[key_samples]
    samples_sem_lo = example[key_samples_sem]
    serialized_example = example['data']

  # Parse sequence example.
  key_input = _RESOLUTIONS[hierarchy_level - 1] + '_' + _INPUT_FEATURE
  key_target = _RESOLUTIONS[hierarchy_level - 1] + '_' + _TARGET_FEATURE
  key_target_sem = _RESOLUTIONS[hierarchy_level - 1] + '_' + _TARGET_SEM_FEATURE
  #key_input = 'input_sdf'
  #key_target = 'target_df'
  #key_target_sem = 'target_sem'

  sequence_features_spec = {
      key_input:
          tf.FixedLenFeature(
              (stored_dim_block, stored_height_block, stored_dim_block, 1),
              tf.float32),
      key_target:
          tf.FixedLenFeature(
              (stored_dim_block, stored_height_block, stored_dim_block, 1),
              tf.float32),
      key_target_sem:
          tf.FixedLenFeature((), tf.string)
  }
  if not is_base_level:
    key_target_lo = _RESOLUTIONS[hierarchy_level] + '_' + _TARGET_FEATURE
    sequence_features_spec[key_target_lo] = tf.FixedLenFeature(
        (stored_dim_block // 2, stored_height_block // 2, stored_dim_block // 2,
         1),
        tf.float32)
    key_target_sem_lo = (
        _RESOLUTIONS[hierarchy_level] + '_' + _TARGET_SEM_FEATURE)
    sequence_features_spec[key_target_sem_lo] = tf.FixedLenFeature((),
                                                                   tf.string)

  example = tf.parse_single_example(serialized_example, sequence_features_spec)
  return key, example, samples_lo, samples_sem_lo
