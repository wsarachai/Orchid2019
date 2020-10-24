from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import initializers, layers
from nets.mobilenet_v2 import default_image_size, create_mobilenet_v2, IMG_SHAPE_224
from stn import pre_spatial_transformer_network


def create_orchid_mobilenet_v2_14(num_classes,
                                  is_training=False,
                                  **kwargs):
    import nets

    if 'step' in kwargs:
        step = kwargs.pop('step')
    else:
        step = ''

    stn_training = False
    branches_training = False

    global_average_layer = keras.layers.GlobalAveragePooling2D()

    data_augmentation = keras.Sequential([
        keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    preprocess_input = keras.applications.mobilenet_v2.preprocess_input

    scales = [0.8, 0.6]

    batch_size = kwargs.pop('batch_size')

    inputs = keras.Input(shape=(default_image_size, default_image_size, 3), batch_size=batch_size)
    inputs = data_augmentation(inputs)
    inputs = preprocess_input(inputs)

    # Create the base model from the pre-trained model MobileNet V2
    stn_base_model = create_mobilenet_v2(input_tensor=inputs,
                                         alpha=1.4,
                                         include_top=False,
                                         weights='imagenet',
                                         sub_name='01')
    branches_base_model = create_mobilenet_v2(input_shape=IMG_SHAPE_224,
                                              alpha=1.4,
                                              include_top=False,
                                              weights='imagenet',
                                              sub_name='02')

    x = stn_base_model(inputs, training=stn_training)
    x = layers.Conv2D(128, [1, 1], activation='relu', name="t1_conv2d_resize_128")(x)
    x = layers.Flatten(name='t1_flatten')(x)
    x = keras.layers.Dropout(rate=0.2, name='t1_dropout')(x)
    x = layers.Dense(128, activation='relu', name='t1_dense_128')(x)

    element_size = 3  # [x, y, scale]
    fc_num = element_size * len(scales)
    h_fc1 = layers.Dense(fc_num, activation='tanh', activity_regularizer='l2', name='t1_dense_6')(x)

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

    # ---- Begin model branches
    all_logits = []
    x = branches_base_model(inputs, training=branches_training)
    all_logits.append(x)
    for i, input_image in enumerate(all_images):
        x = branches_base_model(input_image, training=branches_training)
        all_logits.append(x)

    all_predicts = []
    for i, net in enumerate(all_logits):
        x = global_average_layer(net)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(num_classes, name='t1_fc_{i:02d}'.format(i=i))(x)
        all_predicts.append(x)
    # ---- End model branches

    if step == nets.nets_utils.TRAIN_STEP1:
        outputs = tf.add_n(all_predicts)
        outputs = tf.divide(outputs, len(all_predicts))
    else:
        # estimation block
        c_t = None
        main_net = None
        for i, net in enumerate(all_predicts):
            if i > 0:
                input_and_hstate_concatenated = tf.concat(
                    axis=1,
                    values=[c_t, net],
                    name='t2_concat_{i:02d}'.format(i=i))
                c_t = layers.Dense(num_classes,
                                   kernel_initializer=initializers.truncated_normal(mean=0.5),
                                   name='t2_Dense_{i:02d}'.format(i=i)
                                   )(input_and_hstate_concatenated)
                main_net = main_net + c_t
            else:
                main_net = net
                c_t = net
        outputs = main_net
    model = keras.Model(inputs, outputs)

    if step == nets.nets_utils.TRAIN_STEP1:
        for layer in stn_base_model.layers:
            layer.trainable = False
        for layer in branches_base_model.layers:
            layer.trainable = False
    elif step == nets.nets_utils.TRAIN_STEP2:
        for layer in stn_base_model.layers:
            layer.trainable = False
        for layer in branches_base_model.layers:
            layer.trainable = False
        for layer in model.layers:
            if layer.name.startswith('t2_'):
                layer.trainable = True
            else:
                layer.trainable = False


    model.summary()

    return model
