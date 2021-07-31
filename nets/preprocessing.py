from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops import control_flow_ops
from utils.summary import k_summary


def apply_with_random_selector(x, func, num_cases):
    sel = tf.random.uniform([], maxval=num_cases, dtype=tf.int32)
    return control_flow_ops.merge(
        [func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case) for case in range(num_cases)]
    )[0]


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
    with tf.compat.v1.name_scope(scope, "distort_color", [image]):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
            else:
                raise ValueError("color_ordering must be in [0, 3]")

        # The random_* ops do not necessarily clamp.
        return tf.clip_by_value(image, 0.0, 1.0)


def distorted_bounding_box_crop(
    image,
    bbox,
    min_object_covered=0.1,
    aspect_ratio_range=(0.75, 1.33),
    area_range=(0.05, 1.0),
    max_attempts=100,
    scope=None,
):
    with tf.compat.v1.name_scope(scope, "distorted_bounding_box_crop", [image, bbox]):
        sample_distorted_bounding_box = tf.compat.v1.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bbox,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True,
        )
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        return cropped_image, distort_bbox


def preprocess_for_train(image, height, width, bbox, fast_mode=True, scope=None, add_image_summaries=False):
    with tf.compat.v1.name_scope(scope, "distort_image", [image, height, width, bbox]):
        if bbox is None:
            bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].
        image_with_box = tf.compat.v1.image.draw_bounding_boxes(tf.expand_dims(image, 0), bbox)
        if add_image_summaries:
            _img = image_with_box
            tf.function(k_summary.image_update).get_concrete_function(
                name="image_with_bounding_boxes", unit=_img, max_outputs=1
            )(unit=_img)

        distorted_image, distorted_bbox = distorted_bounding_box_crop(image, bbox)
        # Restore the shape since the dynamic slice based upon the bbox_size loses
        # the third dimension.
        distorted_image.set_shape([None, None, 3])
        image_with_distorted_box = tf.compat.v1.image.draw_bounding_boxes(tf.expand_dims(image, 0), distorted_bbox)
        if add_image_summaries:
            _img = image_with_distorted_box
            tf.function(k_summary.image_update).get_concrete_function(
                name="images_with_distorted_bounding_box", unit=_img, max_outputs=1
            )(unit=_img)

        # This resizing operation may distort the images because the aspect
        # ratio is not respected. We select a resize method in a round robin
        # fashion based on the thread number.
        # Note that ResizeMethod contains 4 enumerated resizing methods.

        # We select only 1 case for fast_mode bilinear.
        num_resize_cases = 1 if fast_mode else 4
        distorted_image = apply_with_random_selector(
            distorted_image,
            lambda x, method: tf.compat.v1.image.resize_images(x, [height, width], method),
            num_cases=num_resize_cases,
        )

        if add_image_summaries:
            _img = image_with_distorted_box
            tf.function(k_summary.image_update).get_concrete_function(
                name="cropped_resized_image", unit=_img, max_outputs=1
            )(unit=_img)

        # Randomly flip the image horizontally.
        distorted_image = tf.compat.v1.image.random_flip_left_right(distorted_image)

        # Randomly distort the colors. There are 1 or 4 ways to do it.
        num_distort_cases = 1 if fast_mode else 4
        distorted_image = apply_with_random_selector(
            distorted_image, lambda x, ordering: distort_color(x, ordering, fast_mode), num_cases=num_distort_cases
        )

        if add_image_summaries:
            _img = image_with_distorted_box
            tf.function(k_summary.image_update).get_concrete_function(
                name="final_distorted_image", unit=_img, max_outputs=1
            )(unit=_img)
        distorted_image = tf.subtract(distorted_image, 0.5)
        distorted_image = tf.multiply(distorted_image, 2.0)
        return distorted_image


def preprocess_for_eval(image, height, width, central_fraction=0.875, scope=None):
    with tf.compat.v1.name_scope(scope, "eval_image", [image, height, width]):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # Crop the central region of the image with an area containing 87.5% of
        # the original image.
        if central_fraction:
            image = tf.image.central_crop(image, central_fraction=central_fraction)

        if height and width:
            # Resize the image to the specified height and width.
            image = tf.expand_dims(image, axis=0)
            image = tf.compat.v1.image.resize_bilinear(image, [height, width], align_corners=False)
            image = tf.squeeze(image, [0])
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image


def preprocess_image(image, height, width, is_training=False, bbox=None, fast_mode=True, add_image_summaries=False):
    if is_training:
        return preprocess_for_train(image, height, width, bbox, fast_mode, add_image_summaries=add_image_summaries)
    else:
        return preprocess_for_eval(image, height, width)
