from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import tensorflow.keras as keras

from nets.core_functions import load_weight_from_old_checkpoint, load_weight
from utils.const import TRAIN_STEP1, TRAIN_STEP2
from utils.lib_utils import get_checkpoint_file


class Orchids52Mobilenet140(object):
    def __init__(self, inputs, outputs, optimizer, loss_fn, mobilenet, predict_layers, training, step):
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

    def compile(self, metrics):
        if self.optimizer and self.loss_fn:
            self.model.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=metrics)
            # run_eagerly=True)

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
        assert self.optimizer is not None and self.predict_layers is not None

        self.checkpoint_path = checkpoint_path
        checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, model=self.model)
        checkpoint_prefix = os.path.join(checkpoint_path, self.step)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, directory=checkpoint_prefix, max_to_keep=self.max_to_keep
        )
        self.checkpoint = (checkpoint, checkpoint_manager)

        predict_layers_path = os.path.join(checkpoint_path, "predict_layers")
        for idx, predict_layer in enumerate(self.predict_layers):
            checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, model=predict_layer)
            prediction_layer_prefix = get_checkpoint_file(predict_layers_path, idx)
            predict_layers_checkpoint_managers = tf.train.CheckpointManager(
                checkpoint, directory=prediction_layer_prefix, max_to_keep=self.max_to_keep
            )
            self.prediction_layer_checkpoints.append((checkpoint, predict_layers_checkpoint_managers))

    def save_model_variables(self):
        _, checkpoint_manager = self.checkpoint
        checkpoint_manager.save()
        for checkpoint in self.prediction_layer_checkpoints:
            _, checkpoint_manager = checkpoint
            checkpoint_manager.save()

    def load_from_v1(
        self, latest_checkpoint, target_model="mobilenetv2_01_1.40_224_", model_name="MobilenetV2", **kwargs
    ):
        return load_weight_from_old_checkpoint(
            latest_checkpoint=latest_checkpoint, target_model=target_model, model_name=model_name
        )

    def restore_model_from_latest_checkpoint_if_exist(self, **kwargs):
        result = False
        show_model_weights = kwargs.get("show_model_weights", False)

        if self.checkpoint:
            checkpoint, checkpoint_manager = self.checkpoint
            if checkpoint_manager.latest_checkpoint:
                status = checkpoint.restore(checkpoint_manager.latest_checkpoint)
                status.assert_existing_objects_matched()
                result = True

        if not result:
            var_loaded = None
            latest_checkpoint = kwargs.pop("checkpoint_path")
            if latest_checkpoint:
                if self.training:
                    if self.step == TRAIN_STEP1:
                        var_loaded = Orchids52Mobilenet140.load_from_v1(
                            self,
                            latest_checkpoint=latest_checkpoint,
                            target_model="mobilenetv2_stn_base_1.40_224_",
                            model_name="MobilenetV2",
                            include_prediction_layer=True,
                        )
                    elif self.step == TRAIN_STEP2:
                        var_loaded = self.load_from_v1(latest_checkpoint=latest_checkpoint, **kwargs)
                else:
                    var_loaded = self.load_from_v1(latest_checkpoint, **kwargs)

                result = load_weight(var_loaded, self.model.weights)

            else:
                result = False
        return result

    def get_step_number_from_latest_checkpoint(self):
        try:
            _, checkpoint_manager = self.checkpoint
            index = checkpoint_manager.latest_checkpoint.index("ckpt-")
            step = checkpoint_manager.latest_checkpoint[index:][5:]
            step = int(step)
        except:
            return 0
        else:
            return step

    def load_weights(self, checkpoint_path):
        self.model.load_weights(checkpoint_path)

    def restore_model_variables(self, load_from_checkpoint_first=True, checkpoint_path=None, **kwargs):
        step = 1
        loaded_successfully = False
        if load_from_checkpoint_first:
            loaded_successfully = self.restore_model_from_latest_checkpoint_if_exist(
                checkpoint_path=checkpoint_path, **kwargs
            )
        if not loaded_successfully:
            self.load_model_variables()
        else:
            step = self.get_step_number_from_latest_checkpoint() + 1
        self.config_layers()
        for var in self.model.trainable_variables:
            print("trainable variable: ", var.name)
        return step

    def config_layers(self):
        if self.step == TRAIN_STEP1:
            self.set_mobilenet_training_status(False)

    def load_model_variables(self):
        if self.step == TRAIN_STEP1:
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

    def save(self, filepath, overwrite=True, include_optimizer=True, save_format=None, signatures=None, options=None):
        model_path = os.path.join(filepath, "model")
        if not tf.io.gfile.exists(model_path):
            tf.io.gfile.makedirs(model_path)
        self.model.save(
            filepath=model_path,
            overwrite=overwrite,
            include_optimizer=include_optimizer,
            save_format=save_format,
            signatures=signatures,
            options=options,
        )
