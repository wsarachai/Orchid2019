from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pathlib
import functools
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


class OrchidsDataset(object):
    def __init__(self):
        self.extracting_label = None

    def _extracting_label(self, file_path, class_names, one_hot):
        if one_hot:
            label = get_label_one_hot(file_path, class_names)
        else:
            label = get_label(file_path, class_names)
        img = tf.io.read_file(file_path)
        return img, label

    def extracting_label_warping(self, data_dir):
        class_names = np.array(sorted([item.name for item in data_dir.glob("n*")]))
        self.extracting_label = wrapped_partial(self._extracting_label, class_names=class_names, one_hot=False)

    def _load_dataset(
        self,
        split,
        root_path,
        batch_size,
        num_of_class,
        train_size,
        test_size,
        validate_size=0,
        repeat=False,
        **kwargs
    ):
        dataset = None
        image_path = os.path.join(root_path, split)
        images_dir = pathlib.Path(image_path)
        dataset = tf.data.Dataset.list_files(str(images_dir / "*/*"), shuffle=False)

        self.extracting_label_warping(data_dir=images_dir)
        decode_dataset = dataset.map(self.extracting_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = configure_for_performance(dataset, batch_size=batch_size)

        if repeat:
            decode_dataset = decode_dataset.repeat()

        if split:
            if split == "train":
                setattr(decode_dataset, "size", train_size)
            elif split == "test":
                setattr(decode_dataset, "size", test_size)
            elif split == "validate":
                setattr(decode_dataset, "size", validate_size)

        setattr(decode_dataset, "num_of_classes", num_of_class)

        return decode_dataset


def load_dataset_v1(split, batch_size, root_path):
    return OrchidsDataset()._load_dataset(
        split=split,
        root_path=root_path,
        batch_size=batch_size,
        num_of_class=orchids52_dataset.NUM_OF_CLASSES,
        train_size=orchids52_dataset.TRAIN_SIZE_V1,
        test_size=orchids52_dataset.TEST_SIZE_V1,
    )


def load_dataset_v2(split, batch_size, root_path):
    return OrchidsDataset()._load_dataset(
        split=split,
        root_path=root_path,
        batch_size=batch_size,
        num_of_class=orchids52_dataset.NUM_OF_CLASSES,
        train_size=orchids52_dataset.TRAIN_SIZE_V2,
        test_size=orchids52_dataset.TEST_SIZE_V2,
        validate_size=orchids52_dataset.VALIDATE_SIZE_V2,
    )
