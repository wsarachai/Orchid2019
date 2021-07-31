from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import layers


def preprocess_image(image, label_values, image_size, central_fraction=0.875, **kwargs):
    # cast_image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # cast_image = tf.image.central_crop(cast_image, central_fraction=central_fraction)
    # image_resize = tf.image.resize(images=cast_image, size=image_size, method=tf.image.ResizeMethod.BILINEAR)

    image_resize = layers.experimental.preprocessing.Resizing(image_size[0], image_size[1])(image)
    # image_resize = layers.experimental.preprocessing.Rescaling(1. / 255)(image_resize)
    # image_resize = tf.subtract(image_resize, 0.5)
    # image_resize = tf.multiply(image_resize, 2.0)

    return image_resize, label_values


NUM_OF_CLASSES = 52
TRAIN_SIZE_V1 = 2820
TEST_SIZE_V1 = 739
TRAIN_SIZE_V2 = 2256
TEST_SIZE_V2 = 739
VALIDATE_SIZE_V2 = 564
TRAIN_SIZE_V3 = 2490
TEST_SIZE_V3 = 855
VALIDATE_SIZE_V3 = 210
