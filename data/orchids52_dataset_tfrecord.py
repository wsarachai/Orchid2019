from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import data
import nets
import tensorflow as tf
from utils.wrapped_tools import wrapped_partial

feature_description = {
    "image/colorspace": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "image/channels": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "image/format": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "image/class/label": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "image/image_raw": tf.io.FixedLenFeature((), tf.string, default_value=""),
}

preprocess_for_train = None
preprocess_for_eval = None


def get_label(serialize_example):
    label = serialize_example["image/class/label"]
    label_string = tf.strings.split(label, ",")
    label_values = tf.strings.to_number(label_string, out_type=tf.dtypes.int64)
    return label_values


def decode_example(serialize_example):
    image = serialize_example["image/image_raw"]
    image = tf.image.decode_jpeg(image, channels=3)
    label_values = get_label(serialize_example)
    return image, label_values


def parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)


def _load_dataset(
    split,
    root_path,
    data_dir,
    batch_size,
    train_size,
    test_size,
    validate_size,
    repeat=False,
    num_readers=1,
    num_map_threads=1,
    **kwargs
):
    pattern = "orchids52-{split}*.tfrecord".format(split=split)
    pattern = os.path.join(root_path, data_dir, pattern)
    dataset = tf.data.Dataset.list_files(file_pattern=pattern)
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        cycle_length=num_readers,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=False,
    )
    parsed_dataset = dataset.map(parse_function, num_parallel_calls=num_map_threads)
    decode_dataset = parsed_dataset.map(decode_example)

    preprocess_image = wrapped_partial(
        data.orchids52_dataset.preprocess_image, image_size=nets.mobilenet_v2.IMG_SIZE_224
    )
    decode_dataset = decode_dataset.map(preprocess_image)
    decode_dataset = decode_dataset.batch(batch_size=batch_size).cache()

    if repeat:
        decode_dataset = decode_dataset.repeat()

    if split:
        if split == "train":
            setattr(decode_dataset, "size", train_size)
        elif split == "test":
            setattr(decode_dataset, "size", test_size)
        elif split == "validate":
            setattr(decode_dataset, "size", validate_size)

    meta_data_path = os.path.join(root_path, data_dir, "orchids52-metadata.txt")
    with open(meta_data_path, "r") as f:
        lines = [line.rstrip().split("\t") for line in f]

    setattr(decode_dataset, "classes", lines)
    setattr(decode_dataset, "num_of_classes", data.orchids52_dataset.NUM_OF_CLASSES)

    return decode_dataset


load_dataset_v2 = wrapped_partial(
    _load_dataset,
    train_size=data.orchids52_dataset.TRAIN_SIZE_V2,
    test_size=data.orchids52_dataset.TEST_SIZE_V2,
    validate_size=data.orchids52_dataset.VALIDATE_SIZE_V2,
    data_dir="tf-records/v2",
)
load_dataset_v3 = wrapped_partial(
    _load_dataset,
    train_size=data.orchids52_dataset.TRAIN_SIZE_V3,
    test_size=data.orchids52_dataset.TEST_SIZE_V3,
    validate_size=data.orchids52_dataset.VALIDATE_SIZE_V3,
    data_dir="tf-records/v3",
)

load_dataset_v2.num_of_class = data.orchids52_dataset.NUM_OF_CLASSES
load_dataset_v2.train_size = data.orchids52_dataset.TRAIN_SIZE_V2
load_dataset_v2.test_size = data.orchids52_dataset.TEST_SIZE_V2
load_dataset_v2.validate_size = data.orchids52_dataset.VALIDATE_SIZE_V2

load_dataset_v3.num_of_class = data.orchids52_dataset.NUM_OF_CLASSES
load_dataset_v3.train_size = data.orchids52_dataset.TRAIN_SIZE_V3
load_dataset_v3.test_size = data.orchids52_dataset.TEST_SIZE_V3
load_dataset_v3.validate_size = data.orchids52_dataset.VALIDATE_SIZE_V3
