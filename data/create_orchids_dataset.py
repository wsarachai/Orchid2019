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
from data.orchids52_dataset import TRAIN_SIZE_V2, TEST_SIZE_V2, TRAIN_SIZE_V1, TEST_SIZE_V1
from data.orchids52_dataset_file import get_label
from lib_utils import start
from nets.mobilenet_v2 import IMG_SIZE_224

ORCHIDS52_DATA_V1 = 'data-v1'
ORCHIDS52_DATA_V2 = 'data-v2'

flags = tf.compat.v1.flags
logging = tf.compat.v1.logging

FLAGS = flags.FLAGS

flags.DEFINE_string('images_dir', '/Volumes/Data/_dataset/_orchids_dataset/orchids52_data',
                    'Original orchid flower images directory')

flags.DEFINE_string('output_directory', '/Volumes/Data/_dataset/_orchids_dataset/orchids52_data/tf-records/v1',
                    'Output data directory')

flags.DEFINE_string('data_version', ORCHIDS52_DATA_V1,
                    'The version of dataset')


def process_path(file_path, class_names, image_size):
    label = get_label(file_path, class_names)
    img = tf.io.read_file(file_path)
    return img, label


def wrapped_partial(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func


def _find_image_files_v1(images_dir, image_size):
    logging.info('Determining list of input files and labels from %s.' % images_dir)
    train_dir = os.path.join(images_dir, 'train-en')
    test_dir = os.path.join(images_dir, 'test-en')

    train_dir = pathlib.Path(train_dir)
    test_dir = pathlib.Path(test_dir)

    train_count = len(list(train_dir.glob('*/*.jpg')))
    test_count = len(list(test_dir.glob('*/*.jpg')))

    train_ds = tf.data.Dataset.list_files(str(train_dir / '*/*'), shuffle=False)
    train_ds = train_ds.shuffle(train_count, reshuffle_each_iteration=True)
    test_ds = tf.data.Dataset.list_files(str(test_dir / '*/*'), shuffle=False)
    test_ds = test_ds.shuffle(test_count, reshuffle_each_iteration=True)
    class_names = np.array(sorted([item.name for item in train_dir.glob('*')]))

    _process_path = wrapped_partial(
        process_path,
        class_names=class_names,
        image_size=image_size)

    train_ds = train_ds.map(_process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.map(_process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_ds = train_ds.take(TRAIN_SIZE_V1)
    test_ds = test_ds.take(TEST_SIZE_V1)

    return train_ds, test_ds


def _find_image_files_v2(images_dir, image_size):
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

    train_ds = all_ds.take(TRAIN_SIZE_V2)
    test_ds = all_ds.skip(TRAIN_SIZE_V2)
    validate_ds = test_ds.skip(TEST_SIZE_V2)
    test_ds = test_ds.take(TEST_SIZE_V2)

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
    train_ds = None
    test_ds = None
    validate_ds = None

    if FLAGS.data_version == ORCHIDS52_DATA_V1:
        train_ds, test_ds = _find_image_files_v1(images_dir=images_dir,
                                                 image_size=image_size)
    elif FLAGS.data_version == ORCHIDS52_DATA_V2:
        train_ds, test_ds, validate_ds = _find_image_files_v2(images_dir=images_dir,
                                                              image_size=image_size)

    if not tf.io.gfile.exists(output_directory):
        tf.io.gfile.mkdir(output_directory)

    if train_ds:
        _write_tf_file(tf_file=os.path.join(output_directory, "orchids52-train.tfrecord"),
                       ds=train_ds)
    if test_ds:
        _write_tf_file(tf_file=os.path.join(output_directory, "orchids52-test.tfrecord"),
                       ds=test_ds)
    if validate_ds:
        _write_tf_file(tf_file=os.path.join(output_directory, "orchids52-validate.tfrecord"),
                       ds=validate_ds)


create_dataset = wrapped_partial(
    _create_dataset,
    image_size=IMG_SIZE_224)


def main(unused_argv):
    create_dataset(FLAGS.images_dir, FLAGS.output_directory)


if __name__ == '__main__':
    start(main)
