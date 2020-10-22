from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.keras as keras

from nets.mobilenet_v2 import create_mobilenet_v2, IMG_SHAPE_224


def create_mobilenet_v2_14(num_classes, freeze_base_model=False, **kwargs):
    # Create the base model from the pre-trained model MobileNet V2
    base_model = create_mobilenet_v2(input_shape=IMG_SHAPE_224,
                                     alpha=1.4,
                                     include_top=False,
                                     weights='imagenet')

    data_augmentation = keras.Sequential([
        keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    preprocess_input = keras.applications.mobilenet_v2.preprocess_input
    global_average_layer = keras.layers.GlobalAveragePooling2D()
    prediction_layer = keras.layers.Dense(num_classes)

    inputs = keras.Input(shape=IMG_SHAPE_224)
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = keras.Model(inputs, outputs)

    if freeze_base_model:
        # Freeze all the layers except for dense layer
        for layer in base_model.layers:
            layer.trainable = False

    model.summary()

    return model
