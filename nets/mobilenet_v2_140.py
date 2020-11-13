from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import nets
import lib_utils
import tensorflow as tf
import tensorflow.keras as keras

from nets.mobilenet_v2 import create_mobilenet_v2, IMG_SHAPE_224


class Orchids52Mobilenet140(object):
    def __init__(self, inputs, outputs,
                 optimizer,
                 loss_fn,
                 base_model,
                 predict_models,
                 training,
                 step):
        super(Orchids52Mobilenet140, self).__init__()
        self.model = keras.Model(inputs, outputs, trainable=training)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.base_model = base_model
        self.predict_models = predict_models
        self.training = training
        self.step = step
        self.max_to_keep = 5

        self.checkpoint = None
        self.checkpoint_manager = None
        self.predict_models_checkpoints = []
        self.predict_models_checkpoint_managers = []

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
        assert (self.optimizer is not None
                and self.predict_models is not None)
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer,
            model=self.model)
        checkpoint_prefix = os.path.join(checkpoint_path, self.step)
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, directory=checkpoint_prefix, max_to_keep=self.max_to_keep)

        predict_models_path = os.path.join(checkpoint_prefix, 'predict_layers')
        for idx, model in enumerate(self.predict_models):
            ck = tf.train.Checkpoint(optimizer=self.optimizer, model=model)
            self.predict_models_checkpoints.append(ck)
            predict_models_checkpoint_prefix = lib_utils.get_checkpoint_file(predict_models_path, idx)
            self.predict_models_checkpoint_managers.append(tf.train.CheckpointManager(
                ck, directory=predict_models_checkpoint_prefix, max_to_keep=self.max_to_keep))

    def save_model_variables(self):
        self.checkpoint_manager.save()
        for p_ck in self.predict_models_checkpoint_managers:
            p_ck.save()

    def restore_model_variables(self):
        latest_checkpoint = self.checkpoint_manager.latest_checkpoint
        if latest_checkpoint:
            status = self.checkpoint.restore(latest_checkpoint)
            status.assert_existing_objects_matched()
            step = int(latest_checkpoint[latest_checkpoint.index('ckpt-'):][5])
            self.config_layers()
            return step
        else:
            self.load_model_variables()
            self.config_layers()
            return 1

    def config_layers(self):
        import nets
        if self.step == nets.utils.TRAIN_STEP1:
            self.set_mobilenet_training_status(False)

    def load_model_variables(self):
        if self.step == nets.utils.TRAIN_STEP1:
            self.load_model_step1()

    def load_model_step1(self):
        for idx, p_ck_mgr in enumerate(self.predict_models_checkpoint_managers):
            if p_ck_mgr.latest_checkpoint:
                status = self.predict_models_checkpoints[idx].restore(p_ck_mgr.latest_checkpoint)
                status.assert_existing_objects_matched()

    def set_mobilenet_training_status(self, trainable):
        if self.base_model:
            self.base_model.trainable = trainable

    def set_prediction_training_status(self, trainable):
        if self.predict_models:
            for p in self.predict_models:
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

    def resume_model_variables(self, filepath, by_name=False, skip_mismatch=False):
        self.config_layers()
        if not hasattr(filepath, 'endswith'):
            filepath = str(filepath)
        self.model.load_weights(
            filepath=filepath, by_name=by_name, skip_mismatch=skip_mismatch)


def create_mobilenet_v2_14(num_classes,
                           optimizer,
                           loss_fn,
                           training=False,
                           **kwargs):
    step = kwargs.pop('step') if 'step' in kwargs else ''
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

    base_model = create_mobilenet_v2(input_shape=IMG_SHAPE_224,
                                     alpha=1.4,
                                     include_top=False,
                                     weights='imagenet')

    x = base_model(x, training=training)
    outputs = prediction_layer(x, training=training)

    model = Orchids52Mobilenet140(inputs, outputs,
                                  optimizer=optimizer,
                                  loss_fn=loss_fn,
                                  base_model=base_model,
                                  predict_models=[prediction_layer],
                                  training=training,
                                  step=step)

    return model
