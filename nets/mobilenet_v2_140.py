from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import h5py
import tensorflow as tf
import tensorflow.keras as keras

from absl import logging

from nets.core_functions import load_weight_from_old_checkpoint
from utils.const import TRAIN_STEP1
from utils.const import TRAIN_TEMPLATE
from utils.lib_utils import get_checkpoint_file


def save_h5_weights(filename, weights):
    f = h5py.File(filename + ".h5", "w")
    try:
        g = f.create_group("model_weights")
        for w in weights:
            val = w.numpy()
            param_dset = g.create_dataset(w.name, val.shape, dtype=val.dtype)
            if not val.shape:
                # scalar
                param_dset[()] = val
            else:
                param_dset[:] = val
    finally:
        f.close()


def save_h5_model_weights(filename, model):
    f = h5py.File(filename + ".h5", "w")
    try:
        g = f.create_group("model_weights")
        for w in model.weights:
            val = w.numpy()
            param_dset = g.create_dataset(w.name, val.shape, dtype=val.dtype)
            if not val.shape:
                # scalar
                param_dset[()] = val
            else:
                param_dset[:] = val
        g = f.create_group("model_optimizer")
        for w in model.optimizer.weights:
            val = w.numpy()
            param_dset = g.create_dataset(w.name, val.shape, dtype=val.dtype)
            if not val.shape:
                # scalar
                param_dset[()] = val
            else:
                param_dset[:] = val
    finally:
        f.close()


def load_from_pretrain1(latest_checkpoint, **kwargs):
    pop_key = kwargs.get("pop_key", True)
    value_to_load = {}
    reader = tf.compat.v1.train.NewCheckpointReader(latest_checkpoint)
    var_to_shape_map = reader.get_variable_to_shape_map()
    key_to_numpy = {}
    for key in sorted(var_to_shape_map.items()):
        print(key)
        key_to_numpy.update({key[0]: reader.get_tensor(key[0])})

    var_maps = {
        "branch_block/prediction_layer/prediction_layer/kernel": "Logits/Conv2d_1c_1x1/weights",
        "branch_block/prediction_layer/prediction_layer/bias": "Logits/Conv2d_1c_1x1/biases",
        "branch_block/prediction_layer_1/prediction_layer/kernel": "Logits/Conv2d_1c_1x1/weights",
        "branch_block/prediction_layer_1/prediction_layer/bias": "Logits/Conv2d_1c_1x1/biases",
        "branch_block/prediction_layer_2/prediction_layer/kernel": "Logits/Conv2d_1c_1x1/weights",
        "branch_block/prediction_layer_2/prediction_layer/bias": "Logits/Conv2d_1c_1x1/biases",
    }

    for var_name in var_maps:
        _key = "MobilenetV2/" + var_maps[var_name]
        if _key in key_to_numpy:
            value = key_to_numpy[_key]
            value_to_load[var_name] = value
            if pop_key:
                key_to_numpy.pop(_key)
        else:
            print("Can't find the key: {}".format(_key))

    if pop_key:
        for key in key_to_numpy:
            print("{} was not loaded".format(key))

    return value_to_load


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
        self.files_may_delete = []

        self.checkpoint_dir = None
        self.checkpoint = None
        self.prediction_layer_checkpoints = []

    def compile(self, metrics):
        if self.optimizer and self.loss_fn:
            self.model.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=metrics)
            # run_eagerly=True)

        #self.model.optimizer._create_all_weights(self.model.weights)

    def process_step(self, inputs, training=False):
        return self.model(inputs, training=training)

    def get_loss(self, labels, predictions):
        return self.loss_fn(labels, predictions)

    def get_regularization_loss(self):
        return self.model.losses

    @property
    def trainable_variables(self):
        return self.model.trainable_variables

    @property
    def variables(self):
        return self.model.variables

    def fit(self, train_ds, initial_epoch, epochs, validation_data=None, callbacks=None):
        return self.model.fit(
            train_ds, initial_epoch=initial_epoch, epochs=epochs, validation_data=validation_data, callbacks=callbacks
        )

    def evaluate(self, validation_data):
        return self.model.evaluate(validation_data)

    def summary(self):
        self.model.summary()

    @property
    def trainable_variables(self):
        return self.model.trainable_variables

    @property
    def variables(self):
        return self.model.variables

    def config_checkpoint(self, checkpoint_dir):
        assert self.optimizer is not None and self.predict_layers is not None

        self.checkpoint_dir = checkpoint_dir
        checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, model=self.model)
        checkpoint_prefix = os.path.join(checkpoint_dir, TRAIN_TEMPLATE.format(self.step))
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, directory=checkpoint_prefix, max_to_keep=self.max_to_keep
        )
        self.checkpoint = (checkpoint, checkpoint_manager)

    def save_model_variables(self):
        _, checkpoint_manager = self.checkpoint
        checkpoint_manager.save()

        # predict_layers_path = os.path.join(self.checkpoint_dir, "predict_layers")
        # for idx, predict_layer in enumerate(self.predict_layers):
        #     if not tf.io.gfile.exists(predict_layers_path):
        #         tf.io.gfile.makedirs(predict_layers_path)
        #     prediction_layer_prefix = get_checkpoint_file(predict_layers_path, idx)
        #     save_h5_weights(prediction_layer_prefix, predict_layer.weights)

        file_to_save = checkpoint_manager.latest_checkpoint + ".h5"
        self.files_may_delete.append(file_to_save)
        save_h5_model_weights(checkpoint_manager.latest_checkpoint, self.model)

        if len(self.files_may_delete) > self.max_to_keep:
            file_to_delete = self.files_may_delete.pop(0)
            tf.io.gfile.remove(file_to_delete)

    def load_from_v1(
        self, latest_checkpoint, target_model="mobilenetv2_01_1.40_224_", model_name="MobilenetV2", **kwargs
    ):
        return load_weight_from_old_checkpoint(
            latest_checkpoint=latest_checkpoint, target_model=target_model, model_name=model_name, **kwargs
        )

    def restore_model_from_latest_checkpoint_if_exist(self, **kwargs):
        result = False
        if self.checkpoint:
            checkpoint, checkpoint_manager = self.checkpoint
            if checkpoint_manager.latest_checkpoint:
                status = checkpoint.restore(checkpoint_manager.latest_checkpoint)
                status.assert_existing_objects_matched()
                result = True

        return result

    def get_step_number_from_latest_checkpoint(self):
        try:
            _, checkpoint_manager = self.checkpoint
            index = checkpoint_manager.latest_checkpoint.index("ckpt-")
            step = checkpoint_manager.latest_checkpoint[index:][5:]
            step = int(step)
        except Exception:
            return 0
        else:
            return step

    def load_weights(self, checkpoint_dir):
        self.model.load_weights(checkpoint_dir)

    def load_model_variables(self):
        pass

    def restore_model_variables(self, checkpoint_dir=None, **kwargs):
        step = 1
        loaded_successfully = self.restore_model_from_latest_checkpoint_if_exist(
            checkpoint_dir=checkpoint_dir, **kwargs
        )
        if loaded_successfully:
            step = self.get_step_number_from_latest_checkpoint() + 1
        else:
            self.load_model_variables()
        return step

    def config_layers(self, **kwargs):
        if self.step == TRAIN_STEP1:
            self.set_mobilenet_training_status(False, **kwargs)

    def set_mobilenet_training_status(self, trainable, fine_tune_at=100):
        if self.mobilenet:
            if trainable:
                self.mobilenet.trainable = True

                if fine_tune_at:
                    # Freeze all the layers before the `fine_tune_at` layer
                    for layer in self.mobilenet.layers[:fine_tune_at]:
                        layer.trainable = False
            else:
                self.mobilenet.trainable = False

    def set_prediction_training_status(self, trainable, **kwargs):
        if self.predict_layers:
            for p in self.predict_layers:
                p.trainable = trainable

    def save(self, filepath, overwrite=True, include_optimizer=True, save_format=None, signatures=None, options=None):
        model_path = os.path.join(filepath, "model")
        if not tf.io.gfile.exists(model_path):
            tf.io.gfile.mkdir(model_path)
        self.model.save(
            filepath=model_path,
            overwrite=overwrite,
            include_optimizer=include_optimizer,
            save_format=save_format,
            signatures=signatures,
            options=options,
        )
