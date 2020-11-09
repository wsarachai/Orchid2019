from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from lib_utils import apply_with_random_selector


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
    return tf.clip_by_value(image, clip_value_min=0, clip_value_max=255)


def _preprocess_for_train(image, label_values, aug_method, image_size):
    method = [tf.image.ResizeMethod.BILINEAR,
              tf.image.ResizeMethod.NEAREST_NEIGHBOR,
              tf.image.ResizeMethod.BICUBIC,
              tf.image.ResizeMethod.LANCZOS3,
              tf.image.ResizeMethod.LANCZOS5,
              tf.image.ResizeMethod.GAUSSIAN,
              tf.image.ResizeMethod.MITCHELLCUBIC]

    cast_image = tf.cast(image, dtype=tf.float32)

    def apply_random_selector(x):
        num_cases = len(method)
        sel = tf.random.uniform([], maxval=num_cases, dtype=tf.int32)
        inputs = [tf.raw_ops.Switch(data=tf.image.resize(images=x,
                                                         size=image_size,
                                                         method=method[case]),
                                    pred=tf.equal(case, sel))[1] for case in range(num_cases)]
        return tf.raw_ops.Merge(inputs=inputs)[0]

    distorted_image = apply_random_selector(cast_image)

    num_distort_cases = 4
    distort_method = True if aug_method == 'fast' else False
    distorted_image = apply_with_random_selector(
        distorted_image,
        lambda x, ordering: distort_color(x, ordering, distort_method),
        num_cases=num_distort_cases)

    flip_image = tf.image.random_flip_left_right(distorted_image)

    return flip_image, label_values


def _preprocess_for_eval(image, label_values, image_size):
    cast_image = tf.cast(image, dtype=tf.float32)
    image_resize = tf.image.resize(images=cast_image,
                                   size=image_size,
                                   method=tf.image.ResizeMethod.BILINEAR)
    return image_resize, label_values


NUM_OF_CLASSES = 52
TRAIN_SIZE_V1 = 2256
TEST_SIZE_V1 = 739
VALIDATE_SIZE_V1 = 564
TRAIN_SIZE_V2 = 2490
TEST_SIZE_V2 = 855
VALIDATE_SIZE_V2 = 210
