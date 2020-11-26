from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pathlib
import functools
import numpy as np
import tensorflow as tf
import nets
from data import orchids52_dataset

logging = tf.compat.v1.logging
already_wrap = False
_process_path = None

preprocess_for_train = None
preprocess_for_eval = None


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
                  aug_method='fast',
                  repeat=False,
                  **kwargs):
    dataset = None
    image_path = os.path.join(root_path, data_dir)
    if 'v1' == data_dir:
        if 'train' == split:
            images_dir = pathlib.Path(os.path.join(image_path, "train-en"))
            dataset = tf.data.Dataset.list_files(str(images_dir / '*/*'), shuffle=False)
        elif 'test' == split:
            images_dir = pathlib.Path(os.path.join(image_path, "test-en"))
            dataset = tf.data.Dataset.list_files(str(images_dir / '*/*'), shuffle=False)
            val_batches = tf.data.experimental.cardinality(dataset)
            dataset = dataset.take(val_batches // 5)
        elif 'validate' == split:
            images_dir = pathlib.Path(os.path.join(image_path, "train-en"))
            dataset = tf.data.Dataset.list_files(str(images_dir / '*/*'), shuffle=False)
            val_batches = tf.data.experimental.cardinality(dataset)
            dataset = dataset.skip(val_batches // 5)

    elif 'v2' == data_dir:
        image_path = pathlib.Path(image_path)
        dataset = tf.data.Dataset.list_files(os.path.join(str(image_path), '*/*'), shuffle=False)
        if 'train' == split:
            dataset = dataset.take(train_size)
        elif 'test' == split:
            dataset = dataset.skip(train_size)
            dataset = dataset.skip(validate_size)
            dataset = dataset.take(test_size)
        elif 'validate' == split:
            dataset = dataset.skip(train_size)
            dataset = dataset.take(validate_size)

    check_wrap_process_path(data_dir=image_path, image_size=nets.mobilenet_v2.IMG_SIZE_224)
    decode_dataset = dataset.map(_process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if split == 'train':
        global preprocess_for_train
        if not preprocess_for_train:
            preprocess_for_train = wrapped_partial(
                orchids52_dataset._preprocess_for_train,
                aug_method=aug_method,
                image_size=nets.mobilenet_v2.IMG_SIZE_224
            )
        decode_dataset = decode_dataset.map(preprocess_for_train)
    else:
        global preprocess_for_eval
        if not preprocess_for_eval:
            preprocess_for_eval = wrapped_partial(
                orchids52_dataset._preprocess_for_eval,
                image_size=nets.mobilenet_v2.IMG_SIZE_224
            )
        decode_dataset = decode_dataset.map(preprocess_for_eval)

    dataset = configure_for_performance(decode_dataset, batch_size=batch_size)

    if repeat:
        dataset = dataset.repeat()

    if split:
        if split == 'train':
            setattr(dataset, 'size', train_size)
        elif split == 'test':
            setattr(dataset, 'size', test_size)
        elif split == 'validate':
            setattr(dataset, 'size', validate_size)

    setattr(dataset, 'num_of_classes', orchids52_dataset.NUM_OF_CLASSES)

    return dataset


load_dataset_v2 = wrapped_partial(
    _load_dataset,
    train_size=orchids52_dataset.TRAIN_SIZE_V2,
    test_size=orchids52_dataset.TEST_SIZE_V2,
    validate_size=orchids52_dataset.VALIDATE_SIZE_V2,
    data_dir='v1')
load_dataset_v3 = wrapped_partial(
    _load_dataset,
    train_size=orchids52_dataset.TRAIN_SIZE_V3,
    test_size=orchids52_dataset.TEST_SIZE_V3,
    validate_size=orchids52_dataset.VALIDATE_SIZE_V3,
    data_dir='v2')

load_dataset_v2.num_of_classes = orchids52_dataset.NUM_OF_CLASSES
load_dataset_v2.train_size = orchids52_dataset.TRAIN_SIZE_V2
load_dataset_v2.test_size = orchids52_dataset.TEST_SIZE_V2
load_dataset_v2.validate_size = orchids52_dataset.VALIDATE_SIZE_V2

load_dataset_v3.num_of_classes = orchids52_dataset.NUM_OF_CLASSES
load_dataset_v3.train_size = orchids52_dataset.TRAIN_SIZE_V3
load_dataset_v3.test_size = orchids52_dataset.TEST_SIZE_V3
load_dataset_v3.validate_size = orchids52_dataset.VALIDATE_SIZE_V3
