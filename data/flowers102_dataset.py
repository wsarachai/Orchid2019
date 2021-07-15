from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def preprocess_image(image, label_values, image_size, central_fraction=0.875, **kwargs):
    cast_image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    cast_image = tf.image.central_crop(cast_image, central_fraction=central_fraction)
    image_resize = tf.image.resize(images=cast_image, size=image_size, method=tf.image.ResizeMethod.BILINEAR)

    image_resize = tf.subtract(image_resize, 0.5)
    image_resize = tf.multiply(image_resize, 2.0)

    return image_resize, label_values


NUM_OF_CLASSES = 102
TRAIN_SIZE_V1 = 6507
TEST_SIZE_V1 = 1682
