from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pathlib
import functools
import h5py
import numpy as np
import tensorflow as tf
from data import orchids52_dataset
from nets.mobilenet_v2 import IMG_SIZE_224


def wrapped_partial(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func


def decode_img(image, size):
    img = tf.image.decode_jpeg(image, channels=3)
    img = tf.image.resize(img, size)
    return img


def get_label_one_hot(file_path, class_names):
    parts = tf.strings.split(file_path, os.path.sep)
    one_hot = parts[-2] == class_names
    return tf.cast(one_hot, tf.float32)


def get_label(file_path, class_names):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]


def configure_for_performance(ds, batch_size=32):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


def _extracting_label(self, file_path, class_names, one_hot):
    if one_hot:
        label = get_label_one_hot(file_path, class_names)
    else:
        label = get_label(file_path, class_names)
    img = tf.io.read_file(file_path)
    return img, label


class OrchidsGenerator(object):
    def __init__(self, split, root_path):
        self.split = split
        self.file = root_path

    def __call__(self, label):
        with h5py.File(self.file, "r") as hf:
            labels = hf["orchids52/" + self.split]
            for im in labels[label]:
                yield im, label


def load_dataset_v1(split, batch_size, root_path):
    file = os.path.join(root_path, split, "orchids52.h5")
    with h5py.File(file, "r") as hf:
        labels = [label for label in hf["orchids52/" + split]]

    ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = ds.interleave(
        lambda label: tf.data.Dataset.from_generator(
            OrchidsGenerator(split=split, root_path=file),
            output_signature=(
                tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32, name=None),
                tf.TensorSpec(shape=(), dtype=tf.string, name=None),
            ),
            args=(label,),
        )
    )
    ds = configure_for_performance(ds, batch_size=batch_size)

    if split:
        if split == "train":
            setattr(ds, "size", orchids52_dataset.TRAIN_SIZE_V1)
        elif split == "test":
            setattr(ds, "size", orchids52_dataset.TEST_SIZE_V1)

    setattr(ds, "num_of_classes", orchids52_dataset.NUM_OF_CLASSES)

    return ds


def load_dataset_v2(split, batch_size, root_path):
    with h5py.File(root_path, "r") as hf:
        labels = [label for label in hf["orchids52/" + split]]

    ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = ds.interleave(
        lambda label: tf.data.Dataset.from_generator(OrchidsGenerator(split=split, root_path=root_path), args=(label,))
    )
    ds = configure_for_performance(ds, batch_size=batch_size)

    if split:
        if split == "train":
            setattr(ds, "size", orchids52_dataset.TRAIN_SIZE_V1)
        elif split == "test":
            setattr(ds, "size", orchids52_dataset.TEST_SIZE_V1)

    setattr(ds, "num_of_classes", orchids52_dataset.NUM_OF_CLASSES)

    return ds
