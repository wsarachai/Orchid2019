from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def distort_color(image, color_ordering=0, fast_mode=True):
    if fast_mode:
        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        else:
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
        elif color_ordering == 2:
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        elif color_ordering == 3:
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            raise ValueError('color_ordering must be in [0, 3]')

    return tf.clip_by_value(image, 0.0, 1.0)


def preprocess_for_train(image, height, width, fast_mode=True):
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    num_resize_cases = 1 if fast_mode else 4
    method = tf.random.uniform([], maxval=num_resize_cases, dtype=tf.int32)
    distorted_image = tf.compat.v1.image.resize_images(image, [height, width], method)
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    ordering = tf.random.uniform([], maxval=num_resize_cases, dtype=tf.int32)
    distorted_image = distort_color(distorted_image, ordering, fast_mode)

    distorted_image = tf.subtract(distorted_image, 0.5)
    distorted_image = tf.multiply(distorted_image, 2.0)
    return distorted_image


def preprocess_for_eval(image, height, width, central_fraction=0.875):
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if central_fraction:
        image = tf.image.central_crop(image, central_fraction=central_fraction)

    if height and width:
        rank = len(image.get_shape().as_list())
        if rank == 3:
            image = tf.expand_dims(image, axis=0)
            image = tf.compat.v1.image.resize_bilinear(image,
                                                       [height, width],
                                                       align_corners=False)
            image = tf.squeeze(image, [0])
        else:
            image = tf.compat.v1.image.resize_bilinear(image,
                                                       [height, width],
                                                       align_corners=False)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


def preprocess_image(image, height, width,
                     is_training=False,
                     fast_mode=True):
    if is_training:
        return preprocess_for_train(image, height, width, fast_mode)
    else:
        return preprocess_for_eval(image, height, width)
