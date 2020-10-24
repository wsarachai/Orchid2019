from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

import tensorflow as tf
from tensorflow.keras import layers

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


def configure_for_performance(ds, batch_size=32):
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    ds = ds.cache()
    return ds


def parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)


def _load_dataset(split,
                  batch_size,
                  data_dir,
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
    normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)
    normalized_ds = parsed_dataset.map(lambda x, y: (normalization_layer(x), y))
    normalized_ds = normalized_ds.batch(batch_size)
    normalized_ds = normalized_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    normalized_ds = normalized_ds.cache()
    return normalized_ds


_decode_example = wrapped_partial(decode_example, image_size=IMG_SIZE_224)
load_dataset = wrapped_partial(
    _load_dataset,
    data_dir='/Volumes/Data/_dataset/_orchids_dataset/orchids52_data/tf-records')
