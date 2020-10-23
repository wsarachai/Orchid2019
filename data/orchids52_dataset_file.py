from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pathlib
import functools
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from nets.mobilenet_v2 import IMG_SIZE_224

logging = tf.compat.v1.logging
already_wrap = False
_process_path = None


def check_wrap_process_path(data_dir, image_size):
    global already_wrap, _process_path
    if not already_wrap:
        class_names = np.array(sorted([item.name for item in data_dir.glob('n*')]))
        _process_path = wrapped_partial(
            process_path,
            class_names=class_names,
            image_size=image_size)
        already_wrap = True


def wrapped_partial(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func


def decode_img(image, size):
    img = tf.image.decode_jpeg(image, channels=3)
    img = tf.image.resize(img, size)
    return img


def get_label(file_path, class_names):
    parts = tf.strings.split(file_path, os.path.sep)
    one_hot = parts[-2] == class_names
    label = tf.cast(one_hot, tf.int8)
    return label


def process_path(file_path, class_names, image_size):
    label = get_label(file_path, class_names)
    img = tf.io.read_file(file_path)
    img = decode_img(img, image_size)
    return img, label


def configure_for_performance(ds, batch_size=32):
    normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)
    ds = ds.map(lambda x, y: (normalization_layer(x), y))
    ds = ds.cache()
    ds = ds.batch(batch_size)
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


def _load_dataset(split,
                  batch_size,
                  data_dir,
                  image_size):
    global _process_path

    if 'train' == split:
        train_data_dir = pathlib.Path(os.path.join(data_dir, "train-en"))
        train_ds = tf.data.Dataset.list_files(str(train_data_dir / '*/*'), shuffle=False)
        check_wrap_process_path(data_dir=train_data_dir, image_size=image_size)
        train_ds = train_ds.map(_process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_ds = configure_for_performance(train_ds, batch_size=batch_size)
        return train_ds
    elif 'test' == split:
        test_data_dir = pathlib.Path(os.path.join(data_dir, "test-en"))
        test_ds = tf.data.Dataset.list_files(str(test_data_dir / '*/*'), shuffle=False)
        val_batches = tf.data.experimental.cardinality(test_ds)
        test_ds = test_ds.take(val_batches // 5)
        check_wrap_process_path(data_dir=test_data_dir, image_size=image_size)
        test_ds = test_ds.map(_process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        test_ds = configure_for_performance(test_ds, batch_size=batch_size)
        return test_ds
    elif 'validate' == split:
        test_data_dir = pathlib.Path(os.path.join(data_dir, "test-en"))
        test_ds = tf.data.Dataset.list_files(str(test_data_dir / '*/*'), shuffle=False)
        val_batches = tf.data.experimental.cardinality(test_ds)
        val_ds = test_ds.skip(val_batches // 5)
        check_wrap_process_path(data_dir=test_data_dir, image_size=image_size)
        val_ds = val_ds.map(_process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val_ds = configure_for_performance(val_ds, batch_size=batch_size)
        return val_ds


load_dataset = wrapped_partial(
    _load_dataset,
    data_dir="/Volumes/Data/_dataset/_orchids_dataset/orchids52_data",
    image_size=IMG_SIZE_224)
