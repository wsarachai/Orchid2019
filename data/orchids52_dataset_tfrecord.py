from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

import tensorflow as tf

from data.orchids52_dataset import TRAIN_SIZE_V1, TEST_SIZE_V1, VALIDATE_SIZE_V2, TRAIN_SIZE_V2, TEST_SIZE_V2, \
    VALIDATE_SIZE_V1
from nets.mobilenet_v2 import IMG_SIZE_224

feature_description = {
    'image/colorspace': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image/channels': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'image/format': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image/class/label': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image/image_raw': tf.io.FixedLenFeature((), tf.string, default_value='')
}


def wrapped_partial(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func


def decode_img(image, size):
    img = tf.image.decode_jpeg(image, channels=3)
    return tf.image.resize(img, size)


def get_label(serialize_example):
    label = serialize_example['image/class/label']
    label_string = tf.strings.split(label, ',')
    label_values = tf.strings.to_number(label_string, out_type=tf.dtypes.int64)
    return label_values


def decode_example(serialize_example, image_size):
    image = serialize_example['image/image_raw']
    image = decode_img(image=image, size=image_size)
    label_values = get_label(serialize_example)
    return image, label_values


def parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)


def _load_dataset(split,
                  data_dir,
                  batch_size,
                  train_size,
                  test_size,
                  validate_size,
                  repeat=False,
                  num_readers=1,
                  num_map_threads=1,
                  **kwargs):
    pattern = "orchids52-{split}*.tfrecord".format(split=split)
    pattern = os.path.join(data_dir, pattern)
    dataset = tf.data.Dataset.list_files(file_pattern=pattern)
    dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x),
                                 cycle_length=num_readers,
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                 deterministic=False)
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = dataset.with_options(
        ignore_order
    )
    parsed_dataset = dataset.map(parse_function, num_parallel_calls=num_map_threads)
    parsed_dataset = parsed_dataset.map(_decode_example)
    parsed_dataset = parsed_dataset.batch(batch_size=batch_size).cache()

    if repeat:
        parsed_dataset = parsed_dataset.repeat()

    if split:
        if split == 'train':
            setattr(parsed_dataset, 'size', train_size)
        elif split == 'test':
            setattr(parsed_dataset, 'size', test_size)
        elif split == 'validate':
            setattr(parsed_dataset, 'size', validate_size)

    return parsed_dataset


_decode_example = wrapped_partial(decode_example, image_size=IMG_SIZE_224)
load_dataset_v1 = wrapped_partial(
    _load_dataset,
    train_size=TRAIN_SIZE_V1,
    test_size=TEST_SIZE_V1,
    validate_size=VALIDATE_SIZE_V1,
    data_dir='/Volumes/Data/_dataset/_orchids_dataset/orchids52_data/tf-records/v1')
load_dataset_v2 = wrapped_partial(
    _load_dataset,
    train_size=TRAIN_SIZE_V2,
    test_size=TEST_SIZE_V2,
    validate_size=VALIDATE_SIZE_V2,
    data_dir='/Volumes/Data/_dataset/_orchids_dataset/orchids52_data/tf-records/v2')
