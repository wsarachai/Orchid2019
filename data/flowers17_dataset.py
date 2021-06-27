from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def preprocess_image(image, label_values, image_size):
    cast_image = tf.cast(image, dtype=tf.float32)
    image_resize = tf.image.resize(images=cast_image,
                                   size=image_size,
                                   method=tf.image.ResizeMethod.BILINEAR)
    return image_resize, label_values


NUM_OF_CLASSES = 17
TRAIN_SIZE_V1 = 1088
TEST_SIZE_V1 = 272
