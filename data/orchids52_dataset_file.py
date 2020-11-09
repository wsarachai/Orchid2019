from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pathlib
import functools
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from data.orchids52_dataset import TRAIN_SIZE_V1, TEST_SIZE_V1, VALIDATE_SIZE_V1, TRAIN_SIZE_V2, TEST_SIZE_V2, \
    VALIDATE_SIZE_V2
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
    return tf.cast(one_hot, tf.float32)


def process_path(file_path, class_names, image_size):
    label = get_label(file_path, class_names)
    img = tf.io.read_file(file_path)
    img = decode_img(img, image_size)
    return img, label


def configure_for_performance(ds, batch_size=32):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


def _load_dataset(split,
                  root_path,
                  data_dir,
                  batch_size,
                  train_size,
                  test_size,
                  validate_size,
                  repeat=False,
                  **kwargs):
    dataset = None
    image_path = os.path.join(root_path, data_dir)
    if 'v1' == data_dir:
        if 'train' == split:
            train_data_dir = pathlib.Path(os.path.join(image_path, "train-en"))
            dataset = tf.data.Dataset.list_files(str(train_data_dir / '*/*'), shuffle=False)
            check_wrap_process_path(data_dir=train_data_dir, image_size=IMG_SIZE_224)
            dataset = dataset.map(_process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = configure_for_performance(dataset, batch_size=batch_size)
        elif 'test' == split:
            test_data_dir = pathlib.Path(os.path.join(image_path, "test-en"))
            dataset = tf.data.Dataset.list_files(str(test_data_dir / '*/*'), shuffle=False)
            val_batches = tf.data.experimental.cardinality(dataset)
            dataset = dataset.take(val_batches // 5)
            check_wrap_process_path(data_dir=test_data_dir, image_size=IMG_SIZE_224)
            dataset = dataset.map(_process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = configure_for_performance(dataset, batch_size=batch_size)
        elif 'validate' == split:
            train_data_dir = pathlib.Path(os.path.join(image_path, "train-en"))
            dataset = tf.data.Dataset.list_files(str(train_data_dir / '*/*'), shuffle=False)
            val_batches = tf.data.experimental.cardinality(dataset)
            dataset = dataset.skip(val_batches // 5)
            check_wrap_process_path(data_dir=train_data_dir, image_size=IMG_SIZE_224)
            dataset = dataset.map(_process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = configure_for_performance(dataset, batch_size=batch_size)

    elif 'v2' == data_dir:
        image_path = pathlib.Path(image_path)
        dataset = tf.data.Dataset.list_files(os.path.join(str(image_path), '*/*'), shuffle=False)
        if 'train' == split:
            dataset = dataset.take(train_size)
            check_wrap_process_path(data_dir=image_path, image_size=IMG_SIZE_224)
            dataset = dataset.map(_process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = configure_for_performance(dataset, batch_size=batch_size)
        elif 'test' == split:
            dataset = dataset.skip(train_size)
            dataset = dataset.skip(validate_size)
            dataset = dataset.take(test_size)
            check_wrap_process_path(data_dir=image_path, image_size=IMG_SIZE_224)
            dataset = dataset.map(_process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = configure_for_performance(dataset, batch_size=batch_size)
        elif 'validate' == split:
            dataset = dataset.skip(train_size)
            dataset = dataset.take(validate_size)
            check_wrap_process_path(data_dir=image_path, image_size=IMG_SIZE_224)
            dataset = dataset.map(_process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = configure_for_performance(dataset, batch_size=batch_size)

    if repeat:
        dataset = dataset.repeat()

    if split:
        if split == 'train':
            setattr(dataset, 'size', train_size)
        elif split == 'test':
            setattr(dataset, 'size', test_size)
        elif split == 'validate':
            setattr(dataset, 'size', validate_size)

    return dataset


load_dataset_v1 = wrapped_partial(
    _load_dataset,
    train_size=TRAIN_SIZE_V1,
    test_size=TEST_SIZE_V1,
    validate_size=VALIDATE_SIZE_V1,
    data_dir='v1')
load_dataset_v2 = wrapped_partial(
    _load_dataset,
    train_size=TRAIN_SIZE_V2,
    test_size=TEST_SIZE_V2,
    validate_size=VALIDATE_SIZE_V2,
    data_dir='v2')
