from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras import initializers
from tensorflow.python.keras import layers

from nets.mobilenet_v2 import default_image_size, create_mobilenet_v2, IMG_SHAPE_224
from stn import pre_spatial_transformer_network


def create_orchid_mobilenet_v2_14(num_classes,
                                  freeze_base_model=False,
                                  is_training=False,
                                  **kwargs):

    global_average_layer = keras.layers.GlobalAveragePooling2D()

    data_augmentation = keras.Sequential([
        keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    preprocess_input = keras.applications.mobilenet_v2.preprocess_input

    scales = [0.8, 0.6]

    batch_size = kwargs.pop('batch_size')

    inputs = keras.Input(shape=(default_image_size, default_image_size, 3), batch_size=batch_size) #batch_input_shape=batch.shape)
    inputs = data_augmentation(inputs)
    inputs = preprocess_input(inputs)

    # Create the base model from the pre-trained model MobileNet V2
    base_model1 = create_mobilenet_v2(input_tensor=inputs,
                                      alpha=1.4,
                                      include_top=False,
                                      weights='imagenet',
                                      sub_name='01')
    base_model2 = create_mobilenet_v2(input_shape=IMG_SHAPE_224,
                                      alpha=1.4,
                                      include_top=False,
                                      weights='imagenet',
                                      sub_name='02')

    x = base_model1(inputs, training=False)
    x = layers.Conv2D(128, [1, 1], activation='relu', name="conv2d_resize_128")(x)
    x = layers.Flatten()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)

    element_size = 3  # [x, y, scale]
    fc_num = element_size * len(scales)
    h_fc1 = layers.Dense(fc_num, activation='tanh', activity_regularizer='l2')(x)

    stn_inputs, bound_err = pre_spatial_transformer_network(inputs,
                                                            h_fc1,
                                                            width=default_image_size,
                                                            height=default_image_size,
                                                            scales=scales)

    if is_training:
        _len = bound_err.get_shape().as_list()[0]
        bound_std = tf.constant(np.full(_len, 0.00, dtype=np.float32))
        tf.losses.mean_squared_error(bound_err, bound_std)

    all_images = []
    for img in stn_inputs:
        all_images.append(img)

    all_logits = []
    x = base_model2(inputs, training=False)
    all_logits.append(x)
    for i, input_image in enumerate(all_images):
        x = base_model2(input_image, training=False)
        all_logits.append(x)

    all_predicts = []
    for i, net in enumerate(all_logits):
        x = global_average_layer(net)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(num_classes)(x)
        all_predicts.append(x)

    if 'step' in kwargs:
        step = kwargs.pop('step')
    else:
        step = ''

    if step == 'pretrain1':
        outputs = tf.add_n(all_predicts)
        outputs = tf.divide(outputs, len(all_predicts))
    else:
        c_t = None
        main_net = None
        for i, net in enumerate(all_predicts):
            if i > 0:
                input_and_hstate_concatenated = tf.concat(axis=1, values=[c_t, net])
                c_t = layers.Dense(num_classes,
                                   kernel_initializer=initializers.truncated_normal(mean=0.5)
                                   )(input_and_hstate_concatenated)
                main_net = main_net + c_t
            else:
                main_net = net
                c_t = net
        outputs = main_net
    model = keras.Model(inputs, outputs)

    if freeze_base_model:
        # Freeze all the layers except for dense layer
        for layer in base_model1.layers:
            layer.trainable = False
        for layer in base_model2.layers:
            layer.trainable = False

    model.summary()

    return model
