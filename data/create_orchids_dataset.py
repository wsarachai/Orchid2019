from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import sys
import pathlib
import tensorflow as tf
import numpy as np
from datetime import datetime
from data.data_utils import _bytes_feature, _int64_feature
from data.orchids52_dataset_file import get_label
from nets.mobilenet_v2 import IMG_SIZE_224

logging = tf.compat.v1.logging


def process_path(file_path, class_names, image_size):
    label = get_label(file_path, class_names)
    img = tf.io.read_file(file_path)
    return img, label


def wrapped_partial(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func


def _find_image_files(images_dir, image_size):
    logging.info('Determining list of input files and labels from %s.' % images_dir)

    all_images_dir = pathlib.Path(images_dir)

    image_count = len(list(all_images_dir.glob('*/*.jpg')))
    all_ds = tf.data.Dataset.list_files(str(all_images_dir / '*/*'), shuffle=False)
    all_ds = all_ds.shuffle(image_count, reshuffle_each_iteration=True)
    class_names = np.array(sorted([item.name for item in all_images_dir.glob('*')]))

    _process_path = wrapped_partial(
        process_path,
        class_names=class_names,
        image_size=image_size)

    all_ds = all_ds.map(_process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_size = int(image_count * 0.7)
    test_size = image_count - train_size
    validate_size = test_size // 5
    test_size = test_size - validate_size

    train_ds = all_ds.take(train_size)
    test_ds = all_ds.skip(train_size)
    validate_ds = test_ds.skip(test_size)
    test_ds = test_ds.take(test_size)

    return train_ds, test_ds, validate_ds


def serialize_example(image_buffer, label):
    colorspace = b'RGB'
    channels = 3
    image_format = b'JPEG'

    features_map = {
        'image/colorspace': _bytes_feature(colorspace),
        'image/channels': _int64_feature(channels),
        'image/format': _bytes_feature(image_format),
        'image/class/label': _bytes_feature(label),
        'image/image_raw': _bytes_feature(image_buffer)
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=features_map))
    return example_proto.SerializeToString()


def _write_tf_file(tf_file, ds):
    parts = tf.strings.split(tf_file, os.path.sep)
    with tf.io.TFRecordWriter(tf_file) as writer:
        num_files = tf.data.experimental.cardinality(ds).numpy()
        for counter, (image, label) in enumerate(ds):
            label = label.numpy()
            label_string = ','.join(str(x) for x in label)
            label_string = label_string.encode('utf8')
            writer.write(serialize_example(image.numpy(), label_string))
            sys.stdout.write('\r>> {} [{}]: Processed {} of {} images in thread batch.'
                             .format(datetime.now(), parts[-1], counter, num_files))
            sys.stdout.flush()


def _create_dataset(images_dir,
                    output_directory,
                    image_size):
    train_ds, test_ds, validate_ds = _find_image_files(images_dir=images_dir,
                                                       image_size=image_size)

    if not tf.io.gfile.exists(output_directory):
        tf.io.gfile.mkdir(output_directory)

    _write_tf_file(tf_file=os.path.join(output_directory, "orchids52-train.tfrecord"),
                   ds=train_ds)
    _write_tf_file(tf_file=os.path.join(output_directory, "orchids52-test.tfrecord"),
                   ds=test_ds)
    _write_tf_file(tf_file=os.path.join(output_directory, "orchids52-validate.tfrecord"),
                   ds=validate_ds)


create_dataset = wrapped_partial(
    _create_dataset,
    image_size=IMG_SIZE_224)
