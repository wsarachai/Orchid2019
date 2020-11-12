from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import nets
import tensorflow.keras as keras

from nets.mobilenet_v2 import create_mobilenet_v2, IMG_SHAPE_224


def create_mobilenet_v2_14(num_classes,
                           training=False,
                           **kwargs):
    freeze_base_model = kwargs.pop('freeze_base_model') if 'freeze_base_model' in kwargs else True
    base_model = create_mobilenet_v2(input_shape=IMG_SHAPE_224,
                                     alpha=1.4,
                                     include_top=False,
                                     weights='imagenet')

    data_augmentation = keras.Sequential([
        keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    preprocess_input = keras.applications.mobilenet_v2.preprocess_input
    prediction_layer = nets.utils.create_predict_module(num_classes=num_classes,
                                                        name='mobilenet_v2_14',
                                                        activation='softmax')

    inputs = keras.Input(shape=IMG_SHAPE_224)
    x = data_augmentation(inputs, training=training)
    x = preprocess_input(x)
    x = base_model(x, training=training)
    outputs = prediction_layer(x, training=training)

    model = keras.Model(inputs, outputs)

    if freeze_base_model:
        for layer in base_model.layers:
            layer.trainable = False
    return model
