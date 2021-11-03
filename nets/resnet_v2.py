from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.applications import imagenet_utils
from nets.const_vars import REGULARIZER_L2, BASE_WEIGHT_PATH, IMG_SHAPE_224
from absl import logging
from utils.const import TRAIN_TEMPLATE
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.util.tf_export import keras_export
from nets import resnet


def ResNet50V2(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    sub_name=None,
):
    """Instantiates the ResNet50V2 architecture."""

    if sub_name:
        name_prefix = "resnetv2_%s_50_224" % (sub_name)
    else:
        name_prefix = "resnetv2_50_224"

    def stack_fn(x):
        x = resnet.stack2(x, 64, 3, name="%s_conv2" % name_prefix)
        x = resnet.stack2(x, 128, 4, name="%s_conv3" % name_prefix)
        x = resnet.stack2(x, 256, 6, name="%s_conv4" % name_prefix)
        return resnet.stack2(x, 512, 3, stride1=1, name="%s_conv5" % name_prefix)

    return resnet.ResNet(
        stack_fn,
        True,
        True,
        "resnet50v2",
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        classifier_activation=classifier_activation,
        name_prefix=name_prefix,
    )


def ResNet101V2(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    """Instantiates the ResNet101V2 architecture."""

    def stack_fn(x):
        x = resnet.stack2(x, 64, 3, name="conv2")
        x = resnet.stack2(x, 128, 4, name="conv3")
        x = resnet.stack2(x, 256, 23, name="conv4")
        return resnet.stack2(x, 512, 3, stride1=1, name="conv5")

    return resnet.ResNet(
        stack_fn,
        True,
        True,
        "resnet101v2",
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        classifier_activation=classifier_activation,
    )


def ResNet152V2(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    """Instantiates the ResNet152V2 architecture."""

    def stack_fn(x):
        x = resnet.stack2(x, 64, 3, name="conv2")
        x = resnet.stack2(x, 128, 8, name="conv3")
        x = resnet.stack2(x, 256, 36, name="conv4")
        return resnet.stack2(x, 512, 3, stride1=1, name="conv5")

    return resnet.ResNet(
        stack_fn,
        True,
        True,
        "resnet152v2",
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        classifier_activation=classifier_activation,
    )


def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(x, data_format=data_format, mode="tf")


def decode_predictions(preds, top=5):
    return imagenet_utils.decode_predictions(preds, top=top)
