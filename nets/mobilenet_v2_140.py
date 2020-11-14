from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import nets
import lib_utils
import tensorflow as tf
import tensorflow.keras as keras


class Orchids52Mobilenet140(object):
    def __init__(self, inputs, outputs,
                 optimizer,
                 loss_fn,
                 mobilenet,
                 predict_layers,
                 training,
                 step):
        super(Orchids52Mobilenet140, self).__init__()
        self.model = keras.Model(inputs, outputs, trainable=training)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.mobilenet = mobilenet
        self.predict_layers = predict_layers
        self.training = training
        self.step = step
        self.max_to_keep = 5

        self.checkpoint_path = None
        self.checkpoint = None
        self.prediction_layer_checkpoints = []

    def compile(self):
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss_fn)

    def process_step(self, inputs, training=False):
        return self.model(inputs, training=training)

    def get_loss(self, labels, predictions):
        return self.loss_fn(labels, predictions)

    def get_regularization_loss(self):
        return self.model.losses

    def get_trainable_variables(self):
        return self.model.trainable_variables

    def summary(self):
        self.model.summary()

    def config_checkpoint(self, checkpoint_path):
        assert (self.optimizer is not None and self.predict_layers is not None)

        self.checkpoint_path = checkpoint_path
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        checkpoint_prefix = os.path.join(checkpoint_path, self.step)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, directory=checkpoint_prefix, max_to_keep=self.max_to_keep)
        self.checkpoint = (checkpoint, checkpoint_manager)

        predict_layers_path = os.path.join(checkpoint_path, 'predict_layers')
        for idx, predict_layer in enumerate(self.predict_layers):
            checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=predict_layer)
            prediction_layer_prefix = lib_utils.get_checkpoint_file(predict_layers_path, idx)
            predict_layers_checkpoint_managers = tf.train.CheckpointManager(
                checkpoint, directory=prediction_layer_prefix, max_to_keep=self.max_to_keep)
            self.prediction_layer_checkpoints.append((checkpoint, predict_layers_checkpoint_managers))

    def save_model_variables(self):
        _, checkpoint_manager = self.checkpoint
        checkpoint_manager.save()
        for checkpoint in self.prediction_layer_checkpoints:
            _, checkpoint_manager = checkpoint
            checkpoint_manager.save()

    def restore_model_from_latest_checkpoint_if_exist(self):
        checkpoint, checkpoint_manager = self.checkpoint
        if checkpoint_manager.latest_checkpoint:
            status = checkpoint.restore(checkpoint_manager.latest_checkpoint)
            status.assert_existing_objects_matched()
            return True
        else:
            return False

    def get_step_number_from_latest_checkpoint(self):
        try:
            _, checkpoint_manager = self.checkpoint
            index = checkpoint_manager.latest_checkpoint.index('ckpt-')
            step = checkpoint_manager.latest_checkpoint[index:][5:]
            step = int(step)
        except NameError:
            return 1
        else:
            return step

    def restore_model_variables(self, load_from_checkpoint_first=True):
        step = 1
        loaded_successfully = False
        if load_from_checkpoint_first:
            loaded_successfully = self.restore_model_from_latest_checkpoint_if_exist()
        if not loaded_successfully:
            self.load_model_variables()
        else:
            step = self.get_step_number_from_latest_checkpoint() + 1
        self.config_layers()
        return step

    def config_layers(self):
        if self.step == nets.utils.TRAIN_STEP1:
            self.set_mobilenet_training_status(False)

    def load_model_variables(self):
        if self.step == nets.utils.TRAIN_STEP1:
            self.load_model_step1()

    def load_model_step1(self):
        latest_checkpoint = None
        for checkpoint, checkpoint_manager in self.prediction_layer_checkpoints:
            if checkpoint_manager.latest_checkpoint:
                # save latest checkpoint, it will reused later on init of pretrain 2 and 3
                latest_checkpoint = checkpoint_manager.latest_checkpoint
            if latest_checkpoint:
                status = checkpoint.restore(latest_checkpoint)
                status.assert_existing_objects_matched()

    def set_mobilenet_training_status(self, trainable):
        if self.mobilenet:
            self.mobilenet.trainable = trainable

    def set_prediction_training_status(self, trainable):
        if self.predict_layers:
            for p in self.predict_layers:
                p.trainable = trainable

    def save(self,
             filepath,
             overwrite=True,
             include_optimizer=True,
             save_format=None,
             signatures=None,
             options=None):
        model_path = os.path.join(filepath, 'model')
        if not tf.io.gfile.exists(model_path):
            tf.io.gfile.mkdir(model_path)
        self.model.save(filepath=model_path,
                        overwrite=overwrite,
                        include_optimizer=include_optimizer,
                        save_format=save_format,
                        signatures=signatures,
                        options=options)


class PreprocessLayer(keras.layers.Layer):
    def __init__(self, mode='horizontal', factor=0.2):
        super(PreprocessLayer, self).__init__()
        self.data_augmentation = keras.Sequential([
            keras.layers.experimental.preprocessing.RandomFlip(mode),
            keras.layers.experimental.preprocessing.RandomRotation(factor),
        ])
        self.preprocess_input = keras.applications.mobilenet_v2.preprocess_input

    def call(self, inputs, **kwargs):
        training = kwargs.pop('training')
        inputs = self.data_augmentation(inputs, training=training)
        inputs = self.preprocess_input(inputs)
        return inputs


class PredictionLayer(keras.layers.Layer):
    def __init__(self, num_classes, activation='linear', dropout_ratio=0.2):
        super(PredictionLayer, self).__init__()
        self.global_average_pooling = keras.layers.GlobalAveragePooling2D()
        self.dropout = keras.layers.Dropout(dropout_ratio)
        self.dense = keras.layers.Dense(num_classes,
                                        activation=activation,
                                        name='dense-{}'.format(num_classes))

    def call(self, inputs, **kwargs):
        training = kwargs.pop('training')
        inputs = self.global_average_pooling(inputs, training=training)
        inputs = self.dropout(inputs, training=training)
        inputs = self.dense(inputs, training=training)
        return inputs


def create_mobilenet_v2_14(num_classes,
                           optimizer,
                           loss_fn,
                           training=False,
                           **kwargs):
    step = kwargs.pop('step') if 'step' in kwargs else nets.utils.TRAIN_TEMPLATE.format(1)

    inputs = keras.Input(shape=nets.mobilenet_v2.IMG_SHAPE_224)
    preprocess_layer = PreprocessLayer()
    mobilenet = nets.mobilenet_v2.create_mobilenet_v2(
        input_shape=nets.mobilenet_v2.IMG_SHAPE_224,
        alpha=1.4,
        include_top=False,
        weights='imagenet')
    prediction_layer = PredictionLayer(num_classes=num_classes,
                                       activation='softmax')

    processed_inputs = preprocess_layer(inputs, training=training)
    mobilenet_logits = mobilenet(processed_inputs, training=training)
    outputs = prediction_layer(mobilenet_logits, training=training)

    model = Orchids52Mobilenet140(inputs, outputs,
                                  optimizer=optimizer,
                                  loss_fn=loss_fn,
                                  mobilenet=mobilenet,
                                  predict_layers=[prediction_layer],
                                  training=training,
                                  step=step)

    return model
