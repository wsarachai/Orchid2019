from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from datetime import datetime
import pathlib
import tensorflow as tf
import numpy as np
import functools

flags = tf.compat.v1.flags
logging = tf.compat.v1.logging

FLAGS = flags.FLAGS

flags.DEFINE_string('images_dir', '/Volumes/Data/_dataset/_orchids_dataset/orchids52_data/all',
                    'Original orchid flower images directory')

flags.DEFINE_string('output_directory', '/Volumes/Data/_dataset/_orchids_dataset/orchids52_data/tf-records',
                    'Output data directory')

default_image_size = 224
IMG_SIZE_224 = (default_image_size, default_image_size)
IMG_SHAPE_224 = IMG_SIZE_224 + (3,)


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def wrapped_partial(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func


def get_label(file_path, class_names):
    parts = tf.strings.split(file_path, os.path.sep)
    one_hot = parts[-2] == class_names
    return tf.cast(one_hot, tf.float32)


def decode_img(img, size):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # resize the image to the desired size
    return tf.image.resize(img, size)


def process_path(file_path, class_names, image_size):
    label = get_label(file_path, class_names)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img, image_size)
    return img, label


def configure_for_performance(ds, batch_size=32):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


def create_dataset(batch_size):
    train_data_dir = pathlib.Path("/Volumes/Data/_dataset/_orchids_dataset/orchids52_data/train-en")
    test_data_dir = pathlib.Path("/Volumes/Data/_dataset/_orchids_dataset/orchids52_data/test-en")

    image_count = len(list(train_data_dir.glob('*/*.jpg')))
    train_ds = tf.data.Dataset.list_files(str(train_data_dir / '*/*'), shuffle=False)
    train_ds = train_ds.shuffle(image_count, reshuffle_each_iteration=False)
    test_ds = tf.data.Dataset.list_files(str(test_data_dir / '*/*'), shuffle=False)

    class_names = np.array(sorted([item.name for item in train_data_dir.glob('*')]))
    print(class_names)

    val_batches = tf.data.experimental.cardinality(test_ds)
    val_ds = test_ds.skip(val_batches // 5)
    test_ds = test_ds.take(val_batches // 5)

    print(tf.data.experimental.cardinality(train_ds).numpy())
    print(tf.data.experimental.cardinality(val_ds).numpy())
    print(tf.data.experimental.cardinality(test_ds).numpy())

    _process_path = wrapped_partial(
        process_path,
        class_names=class_names,
        image_size=IMG_SIZE_224)

    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    train_ds = train_ds.map(_process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(_process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.map(_process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_ds = configure_for_performance(train_ds, batch_size=batch_size)
    val_ds = configure_for_performance(val_ds, batch_size=batch_size)
    test_ds = configure_for_performance(test_ds, batch_size=batch_size)

    num_classes = len(class_names)

    return train_ds, val_ds, test_ds, num_classes


def _find_image_files(images_dir):
    logging.info('Determining list of input files and labels from %s.' % images_dir)

    all_images_dir = pathlib.Path(images_dir)

    image_count = len(list(all_images_dir.glob('*/*.jpg')))
    all_ds = tf.data.Dataset.list_files(str(all_images_dir / '*/*'), shuffle=False)
    all_ds = all_ds.shuffle(image_count, reshuffle_each_iteration=True)
    class_names = np.array(sorted([item.name for item in all_images_dir.glob('*')]))

    _process_path = wrapped_partial(
        process_path,
        class_names=class_names,
        image_size=IMG_SIZE_224)

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


def _convert_to_example(image_buffer,
                        label):
    colorspace = b'RGB'
    channels = 3
    image_format = b'JPEG'

    features_map = {
        'image/colorspace': _bytes_feature(colorspace),
        'image/channels': _int64_feature(channels),
        'image/format': _bytes_feature(image_format),
        'image/class/label': _int64_feature(label),
        'image/image_raw': _bytes_feature(image_buffer)
    }

    return tf.train.Example(features=tf.train.Features(feature=features_map))


def _process_dataset(images_dir,
                     output_directory):
    train_ds, test_ds, validate_ds = _find_image_files(images_dir)

    if not tf.io.gfile.exists(output_directory):
        tf.io.gfile.mkdir(output_directory)

    train_file = os.path.join(output_directory, "orchids52-train.tfrecord")

    with tf.io.TFRecordWriter(train_file) as writer:
        num_files = tf.data.experimental.cardinality(train_ds).numpy()
        for counter, (image, label) in enumerate(train_ds):
            example = _convert_to_example(image_buffer=image, label=label)
            writer.write(example.SerializeToString())
            if not counter % 8:
                sys.stdout.write('\r>> {} [train]: Processed {} of {} images in thread batch.'
                                 .format(datetime.now(), counter, num_files))
                sys.stdout.flush()


def main(unused_argv):
    logging.info('Saving results to %s' % FLAGS.output_directory)

    _process_dataset(FLAGS.images_dir, FLAGS.output_directory)


if __name__ == '__main__':
    print("tf.version %s" % tf.__version__)
    logging.set_verbosity(logging.INFO)
    tf.compat.v1.app.run(main)
