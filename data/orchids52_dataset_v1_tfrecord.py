from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from data import orchids52_dataset
from utils.wrapped_tools import wrapped_partial

feature_description = {
    "image/height": tf.io.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    "image/width": tf.io.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    "image/colorspace": tf.io.FixedLenFeature((), tf.string, default_value=""),
    "image/channels": tf.io.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    "image/class/label": tf.io.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    "image/class/synset": tf.io.FixedLenFeature((), tf.string, default_value=""),
    "image/class/text": tf.io.FixedLenFeature((), tf.string, default_value=""),
    "image/format": tf.io.FixedLenFeature((), tf.string, default_value=""),
    "image/filename": tf.io.FixedLenFeature((), tf.string, default_value=""),
    "image/encoded": tf.io.FixedLenFeature((), tf.string, default_value=""),
    "bottleneck/inception_v1": tf.io.FixedLenFeature((), tf.string, default_value=""),
    "bottleneck/inception_v3": tf.io.FixedLenFeature((), tf.string, default_value=""),
}


def _get_label(serialize_example, depth, one_hot=False):
    label = serialize_example["image/class/label"]
    if one_hot:
        label_values = tf.one_hot(label, depth=depth)
        tf.assert_equal(tf.argmax(label_values), label)
    else:
        label_values = label
    return label_values


def decode_example(serialize_example):
    image = serialize_example["image/encoded"]
    image = tf.image.decode_jpeg(image, channels=3)
    label_values = get_label(serialize_example)
    return image, label_values


def decode_example_one_hot(serialize_example):
    image = serialize_example["image/encoded"]
    image = tf.image.decode_jpeg(image, channels=3)
    label_values = get_label_one_hot(serialize_example)
    return image, label_values


def parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)


def _load_dataset(split, root_path, train_size, test_size, repeat=False, num_readers=1, num_map_threads=1, **kwargs):
    one_hot = kwargs.get("one_hot", False)
    pattern = "orchids52_{split}*.tfrecord".format(split=split)
    pattern = os.path.join(root_path, pattern)
    dataset = tf.data.Dataset.list_files(file_pattern=pattern)
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        cycle_length=num_readers,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=False,
    )
    parsed_dataset = dataset.map(parse_function, num_parallel_calls=num_map_threads)

    if one_hot:
        decode_dataset = parsed_dataset.map(decode_example_one_hot)
    else:
        decode_dataset = parsed_dataset.map(decode_example)

    if split == "train":
        decode_dataset = decode_dataset.shuffle(280, reshuffle_each_iteration=True)

    # preprocess_image = wrapped_partial(orchids52_dataset.preprocess_image, image_size=IMG_SIZE_224)
    # decode_dataset = decode_dataset.map(preprocess_image)
    # decode_dataset = decode_dataset.batch(batch_size=batch_size)
    # decode_dataset = decode_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    if repeat:
        decode_dataset = decode_dataset.repeat()

    if split == "train":
        setattr(decode_dataset, "size", train_size)
    elif split == "test":
        setattr(decode_dataset, "size", test_size)

    meta_data_path = os.path.join(root_path, "orchids52_metadata.txt")
    with open(meta_data_path, "r") as f:
        lines = [line.rstrip().split("\t") for line in f]

    setattr(decode_dataset, "classes", lines)
    setattr(decode_dataset, "num_of_classes", orchids52_dataset.NUM_OF_CLASSES)

    return decode_dataset


get_label = wrapped_partial(_get_label, depth=orchids52_dataset.NUM_OF_CLASSES)
get_label_one_hot = wrapped_partial(_get_label, depth=orchids52_dataset.NUM_OF_CLASSES, one_hot=True)

load_dataset_v1 = wrapped_partial(
    _load_dataset, train_size=orchids52_dataset.TRAIN_SIZE_V1, test_size=orchids52_dataset.TEST_SIZE_V1
)

load_dataset_v1.num_of_classes = orchids52_dataset.NUM_OF_CLASSES
load_dataset_v1.train_size = orchids52_dataset.TRAIN_SIZE_V1
load_dataset_v1.test_size = orchids52_dataset.TEST_SIZE_V1
