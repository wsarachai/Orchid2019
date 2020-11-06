from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

import tensorflow as tf

from data.orchids52_dataset import TRAIN_SIZE_V1, TEST_SIZE_V1, VALIDATE_SIZE_V2, TRAIN_SIZE_V2, TEST_SIZE_V2, \
    VALIDATE_SIZE_V1
from lib_utils import apply_with_random_selector
from nets.mobilenet_v2 import IMG_SIZE_224

feature_description = {
    'image/colorspace': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image/channels': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'image/format': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image/class/label': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image/image_raw': tf.io.FixedLenFeature((), tf.string, default_value='')
}
preprocess_for_train = None


def wrapped_partial(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func


def get_label(serialize_example):
    label = serialize_example['image/class/label']
    label_string = tf.strings.split(label, ',')
    label_values = tf.strings.to_number(label_string, out_type=tf.dtypes.int64)
    return label_values


def decode_example(serialize_example):
    image = serialize_example['image/image_raw']
    image = tf.image.decode_jpeg(image, channels=3)
    label_values = get_label(serialize_example)
    return image, label_values


def parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)


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


def _preprocess_for_train(image, label_values, aug_method):
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
                                                         size=IMG_SIZE_224,
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


def preprocess_for_eval(image, label_values):
    cast_image = tf.cast(image, dtype=tf.float32)
    image_resize = tf.image.resize(images=cast_image,
                                   size=IMG_SIZE_224,
                                   method=tf.image.ResizeMethod.BILINEAR)
    return image_resize, label_values


def _load_dataset(split,
                  root_path,
                  data_dir,
                  batch_size,
                  train_size,
                  test_size,
                  validate_size,
                  repeat=False,
                  aug_method='fast',
                  num_readers=1,
                  num_map_threads=1,
                  **kwargs):
    pattern = "orchids52-{split}*.tfrecord".format(split=split)
    pattern = os.path.join(root_path, data_dir, pattern)
    dataset = tf.data.Dataset.list_files(file_pattern=pattern)
    dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x),
                                 cycle_length=num_readers,
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                 deterministic=False)
    parsed_dataset = dataset.map(parse_function, num_parallel_calls=num_map_threads)
    decode_dataset = parsed_dataset.map(decode_example)

    if split == 'train':
        global preprocess_for_train
        if not preprocess_for_train:
            preprocess_for_train = wrapped_partial(
                _preprocess_for_train,
                aug_method=aug_method
            )
        decode_dataset = decode_dataset.map(preprocess_for_train)
    else:
        decode_dataset = decode_dataset.map(preprocess_for_eval)

    decode_dataset = decode_dataset.batch(batch_size=batch_size).cache()

    if repeat:
        decode_dataset = decode_dataset.repeat()

    if split:
        if split == 'train':
            setattr(decode_dataset, 'size', train_size)
        elif split == 'test':
            setattr(decode_dataset, 'size', test_size)
        elif split == 'validate':
            setattr(decode_dataset, 'size', validate_size)

    return decode_dataset


load_dataset_v1 = wrapped_partial(
    _load_dataset,
    train_size=TRAIN_SIZE_V1,
    test_size=TEST_SIZE_V1,
    validate_size=VALIDATE_SIZE_V1,
    data_dir='tf-records/v1')
load_dataset_v2 = wrapped_partial(
    _load_dataset,
    train_size=TRAIN_SIZE_V2,
    test_size=TEST_SIZE_V2,
    validate_size=VALIDATE_SIZE_V2,
    data_dir='tf-records/v2')
