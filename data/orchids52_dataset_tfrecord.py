from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

import tensorflow as tf

from data.orchids52_dataset import TRAIN_SIZE_V1, TEST_SIZE_V1, VALIDATE_SIZE_V2, TRAIN_SIZE_V2, TEST_SIZE_V2, \
    VALIDATE_SIZE_V1, _preprocess_for_train, _preprocess_for_eval
from nets.mobilenet_v2 import IMG_SIZE_224

feature_description = {
    'image/colorspace': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image/channels': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'image/format': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image/class/label': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image/image_raw': tf.io.FixedLenFeature((), tf.string, default_value='')
}

preprocess_for_train = None
preprocess_for_eval = None


def wrapped_partial(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func


def get_label(serialize_example):
    label = serialize_example['image/class/label']
    label_string = tf.strings.split(label, ',')
    label_values = tf.strings.to_number(label_string, out_type=tf.dtypes.int64)
    return label_values


def decode_example(serialize_example):
    image = serialize_example['image/image_raw']
    image = tf.image.decode_jpeg(image, channels=3)
    label_values = get_label(serialize_example)
    return image, label_values


def parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)


def _load_dataset(split,
                  root_path,
                  data_dir,
                  batch_size,
                  train_size,
                  test_size,
                  validate_size,
                  repeat=False,
                  aug_method='fast',
                  num_readers=1,
                  num_map_threads=1,
                  **kwargs):
    pattern = "orchids52-{split}*.tfrecord".format(split=split)
    pattern = os.path.join(root_path, data_dir, pattern)
    dataset = tf.data.Dataset.list_files(file_pattern=pattern)
    dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x),
                                 cycle_length=num_readers,
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                 deterministic=False)
    parsed_dataset = dataset.map(parse_function, num_parallel_calls=num_map_threads)
    decode_dataset = parsed_dataset.map(decode_example)

    if split == 'train':
        global preprocess_for_train
        if not preprocess_for_train:
            preprocess_for_train = wrapped_partial(
                _preprocess_for_train,
                aug_method=aug_method,
                image_size=IMG_SIZE_224
            )
        decode_dataset = decode_dataset.map(preprocess_for_train)
    else:
        global preprocess_for_eval
        if not preprocess_for_eval:
            preprocess_for_eval = wrapped_partial(
                _preprocess_for_eval,
                image_size=IMG_SIZE_224
            )
        decode_dataset = decode_dataset.map(preprocess_for_eval)

    decode_dataset = decode_dataset.batch(batch_size=batch_size).cache()

    if repeat:
        decode_dataset = decode_dataset.repeat()

    if split:
        if split == 'train':
            setattr(decode_dataset, 'size', train_size)
        elif split == 'test':
            setattr(decode_dataset, 'size', test_size)
        elif split == 'validate':
            setattr(decode_dataset, 'size', validate_size)

    return decode_dataset


load_dataset_v1 = wrapped_partial(
    _load_dataset,
    train_size=TRAIN_SIZE_V1,
    test_size=TEST_SIZE_V1,
    validate_size=VALIDATE_SIZE_V1,
    data_dir='tf-records/v1')
load_dataset_v2 = wrapped_partial(
    _load_dataset,
    train_size=TRAIN_SIZE_V2,
    test_size=TEST_SIZE_V2,
    validate_size=VALIDATE_SIZE_V2,
    data_dir='tf-records/v2')
