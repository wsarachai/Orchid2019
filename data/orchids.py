from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

logging = tf.compat.v1.logging

default_image_size = 224
IMG_SIZE_224 = (default_image_size, default_image_size)
IMG_SHAPE_224 = IMG_SIZE_224 + (3,)


def decode_img(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    #image = tf.reshape(image, [*size, 3])
    return image


def _parse_function(example_proto):
    feature_description = {
        'image/colorspace': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/channels': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'image/format': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/class/label': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/image_raw': tf.io.FixedLenFeature((), tf.string, default_value='')
    }
    return tf.io.parse_single_example(example_proto, feature_description)


def load_dataset(filename):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    filenames = [filename]
    raw_dataset = tf.data.TFRecordDataset(filenames)
    raw_dataset = raw_dataset.with_options(
        ignore_order
    )
    return raw_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
