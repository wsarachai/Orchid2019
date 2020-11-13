from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import functools
import nets
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras import Sequential
from nets.mobilenet_v2 import default_image_size, create_mobilenet_v2, IMG_SHAPE_224
from stn import pre_spatial_transformer_network

logging = tf.compat.v1.logging


class Orchids52Mobilenet140STN(nets.mobilenet_v2_140.Orchids52Mobilenet140):
    def __init__(self, inputs, outputs,
                 optimizer,
                 loss_fn,
                 base_model,
                 stn_dense,
                 estimate_block,
                 predict_models,
                 branch_model,
                 boundary_loss,
                 training,
                 step):
        super(Orchids52Mobilenet140STN, self).__init__(inputs, outputs,
                                                       optimizer,
                                                       loss_fn,
                                                       base_model,
                                                       predict_models,
                                                       training,
                                                       step)
        self.stn_dense = stn_dense
        self.branch_model = branch_model
        self.estimate_block = estimate_block
        self.boundary_loss = boundary_loss
        self.stn_dense_checkpoint = None
        self.stn_dense_checkpoint_manager = None

    def config_checkpoint(self, checkpoint_path):
        super(Orchids52Mobilenet140STN, self).config_checkpoint(checkpoint_path)
        if self.stn_dense:
            self.stn_dense_checkpoint = tf.train.Checkpoint(
                optimizer=self.optimizer,
                model=self.stn_dense)
            checkpoint_prefix = os.path.join(checkpoint_path, self.step)
            stn_dense_models_path = os.path.join(checkpoint_prefix, 'stn_dense_layers')
            self.stn_dense_checkpoint_manager = tf.train.CheckpointManager(
                self.stn_dense_checkpoint,
                directory=stn_dense_models_path,
                max_to_keep=self.max_to_keep)

    def save_model_variables(self):
        super(Orchids52Mobilenet140STN, self).save_model_variables()
        if self.stn_dense:
            self.stn_dense_checkpoint.save()

    def set_mobilenet_training_status(self, trainable):
        super(Orchids52Mobilenet140STN, self).set_mobilenet_training_status(trainable)
        if self.branch_model:
            self.branch_model.trainable = trainable

    def config_layers(self):
        import nets
        if self.step == nets.utils.TRAIN_STEP1:
            self.set_mobilenet_training_status(False)
        elif self.step == nets.utils.TRAIN_STEP2:
            self.set_mobilenet_training_status(False)
            self.set_prediction_training_status(False)
        elif self.step == nets.utils.TRAIN_STEP3:
            self.set_mobilenet_training_status(False)
            self.set_prediction_training_status(False)
            self.stn_dense.trainable = False
        elif self.step == nets.utils.TRAIN_V2_STEP1:
            self.set_mobilenet_training_status(False)
        elif self.step == nets.utils.TRAIN_V2_STEP2:
            self.set_mobilenet_training_status(True)

    def load_model_step2(self):
        self.load_model_step1()
        if self.stn_dense_checkpoint_manager.latest_checkpoint:
            status = self.stn_dense_checkpoint.restore(
                self.stn_dense_checkpoint_manager.latest_checkpoint)
            status.assert_existing_objects_matched()

    def load_model_step3(self):
        latest_checkpoint = self.checkpoint_manager.latest_checkpoint
        if latest_checkpoint:
            status = self.checkpoint.restore(latest_checkpoint)
            status.assert_existing_objects_matched()

    def load_model_step4(self):
        self.load_model_step3()

    def load_model_v2_step2(self):
        pass

    def load_model_variables(self):
        if self.step == nets.utils.TRAIN_STEP1:
            self.load_model_step1()
        elif self.step == nets.utils.TRAIN_STEP2:
            self.load_model_step2()
        elif self.step == nets.utils.TRAIN_STEP3:
            self.load_model_step3()
        elif self.step == nets.utils.TRAIN_STEP4:
            self.load_model_step4()
        elif self.step == nets.utils.TRAIN_V2_STEP2:
            self.load_model_v2_step2()


class BranchesBlock(keras.layers.Layer):
    def __init__(self, num_classes):
        super(BranchesBlock, self).__init__()
        self.branch_base_model = create_mobilenet_v2(input_shape=IMG_SHAPE_224,
                                                     alpha=1.4,
                                                     include_top=False,
                                                     weights='imagenet',
                                                     sub_name='02')
        self.branches_prediction_models = [
            nets.utils.create_predict_module(
                num_classes=num_classes,
                name='prediction_layer-1',
                activation='softmax'),
            nets.utils.create_predict_module(
                num_classes=num_classes,
                name='prediction_layer-2',
                activation='softmax'),
            nets.utils.create_predict_module(
                num_classes=num_classes,
                name='prediction_layer-3',
                activation='softmax')
        ]

    def call(self, inputs, **kwargs):
        logits = tf.convert_to_tensor([
            self.sub_process(inputs[i], i) for i in range(0, 3)
        ], name='logits')
        return logits

    def sub_process(self, inp, i):
        x = self.branch_base_model(inp, training=False)
        return self.branches_prediction_models[i](x, training=False)


class EstimationBlock(keras.layers.Layer):
    def __init__(self, num_classes, batch_size):
        super(EstimationBlock, self).__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.dense1 = keras.layers.Dense(num_classes, name='t2_Dense_1')
        self.dense2 = keras.layers.Dense(num_classes, name='t2_Dense_2')

    def call(self, inputs, **kwargs):
        inp1 = tf.slice(inputs, [0, 0, 0], [1, self.batch_size, self.num_classes])
        inp2 = tf.slice(inputs, [1, 0, 0], [1, self.batch_size, self.num_classes])
        inp3 = tf.slice(inputs, [2, 0, 0], [1, self.batch_size, self.num_classes])
        inp1 = tf.squeeze(inp1)
        inp2 = tf.squeeze(inp2)
        inp3 = tf.squeeze(inp3)

        main_net = c_t = inp1

        input_and_hstate_concatenated = tf.concat(values=[c_t, inp2], axis=1)
        c_t = self.dense1(input_and_hstate_concatenated)
        main_net = main_net + c_t

        input_and_hstate_concatenated = tf.concat(values=[c_t, inp3], axis=1)
        c_t = self.dense2(input_and_hstate_concatenated)
        main_net = main_net + c_t

        return main_net


def create_orchid_mobilenet_v2_14(num_classes,
                                  optimizer,
                                  loss_fn,
                                  training=False,
                                  **kwargs):
    stn_dense = None
    branch_base_model = None
    boundary_loss = None
    estimate_block = None
    branches_prediction_models = []
    step = kwargs.pop('step') if 'step' in kwargs else ''
    batch_size = kwargs.pop('batch_size') if 'batch_size' in kwargs else 32
    model_name = 'orchids52-en'

    with tf.Graph().as_default():
        with tf.name_scope(model_name):
            data_augmentation = keras.Sequential([
                keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
                keras.layers.experimental.preprocessing.RandomRotation(0.2),
            ], name='data-augmentation')
            preprocess_input = keras.applications.mobilenet_v2.preprocess_input

            inputs = keras.Input(shape=IMG_SHAPE_224, name='input_224_224_3', batch_size=batch_size)
            aug_inputs = data_augmentation(inputs, training=training)
            preprocess_inputs = preprocess_input(aug_inputs)

            stn_base_model = create_mobilenet_v2(input_shape=IMG_SHAPE_224,
                                                 alpha=1.4,
                                                 include_top=False,
                                                 weights='imagenet',
                                                 sub_name='01')

            if step != nets.utils.TRAIN_STEP1:
                with tf.name_scope('stn'):
                    scales = [0.8, 0.6]
                    element_size = 3  # [x, y, scale]
                    fc_num = element_size * len(scales)

                    stn_dense = Sequential([
                        keras.layers.Conv2D(128, [1, 1], activation='relu', name="t2_stn_conv2d_resize_128"),
                        keras.layers.Flatten(name='t2_stn_flatten'),
                        keras.layers.Dense(128, name='t2_stn_dense_128'),
                        keras.layers.Dropout(rate=0.2, name='t2_stn_dropout'),
                        keras.layers.Dense(fc_num, activation='tanh', activity_regularizer='l2', name='t1_dense_6')
                    ], name='stn_dense')

                    localization_network = Sequential([
                        stn_base_model,
                        stn_dense
                    ], name='localization_network')

                    loc_output = localization_network(preprocess_inputs)

                    with tf.name_scope('transformer_network'):
                        stn_output, bound_err = pre_spatial_transformer_network(preprocess_inputs,
                                                                                loc_output,
                                                                                batch_size=batch_size,
                                                                                width=default_image_size,
                                                                                height=default_image_size,
                                                                                scales=scales)

                        if training:
                            with tf.name_scope('boundary_loss'):
                                _len = bound_err.shape[0]
                                bound_std = tf.constant(np.full(_len, 0.00, dtype=np.float32),
                                                        name='bound_std')
                                boundary_loss = keras.Model(inputs,
                                                            tf.keras.losses.MSE(bound_err, bound_std),
                                                            name='mse')
                    stn_output = tf.stack([inputs, stn_output[0], stn_output[1]], axis=0)

                with tf.name_scope('branches'):
                    branches_block = BranchesBlock(num_classes=num_classes)
                    branch_base_model = branches_block.branch_base_model
                    branches_prediction_models = branches_block.branches_prediction_models

                    logits = branches_block(stn_output)

                with tf.name_scope('estimate_block'):
                    if step == nets.utils.TRAIN_STEP2:
                        outputs = tf.reduce_mean(logits, axis=0)
                    else:
                        estimate_block = EstimationBlock(num_classes=num_classes, batch_size=batch_size)
                        outputs = estimate_block(logits)

            else:
                prediction_layer = nets.utils.create_predict_module(num_classes=num_classes,
                                                                    activation='softmax')
                branches_prediction_models.append(prediction_layer)
                stn_output = stn_base_model(preprocess_inputs, training=training)
                outputs = prediction_layer(stn_output, training=training)

            model = Orchids52Mobilenet140STN(inputs, outputs,
                                             optimizer=optimizer,
                                             loss_fn=loss_fn,
                                             base_model=stn_base_model,
                                             stn_dense=stn_dense,
                                             estimate_block=estimate_block,
                                             predict_models=branches_prediction_models,
                                             branch_model=branch_base_model,
                                             boundary_loss=boundary_loss,
                                             training=training,
                                             step=step)
            return model


def _handle_boundary_loss(name, variable, error_fn):
    def _loss_for_boundary(v):
        with tf.name_scope(name + '/boundary_loss'):
            regularization = error_fn(v)
        return regularization

    return functools.partial(_loss_for_boundary, variable)
