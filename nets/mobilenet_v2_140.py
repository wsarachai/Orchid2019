from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import nets
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.python.keras import activations
from nets.mobilenet_v2 import IMG_SHAPE_224
from nets.mobilenet_v2 import create_mobilenet_v2
from utils.const import TRAIN_STEP1, TRAIN_STEP2, TRAIN_STEP4
from utils.const import TRAIN_TEMPLATE
from utils.lib_utils import get_checkpoint_file


def preprocess_input(image_data, central_fraction=0.875):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.central_crop(image, central_fraction=central_fraction)
    image = tf.image.resize(images=image, size=nets.mobilenet_v2.IMG_SIZE_224, method=tf.image.ResizeMethod.BILINEAR)

    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)

    return tf.expand_dims(image, axis=0)


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

        self.checkpoint_dir = None
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

    def config_checkpoint(self, checkpoint_dir):
        assert self.optimizer is not None and self.predict_layers is not None

        self.checkpoint_dir = checkpoint_dir
        checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, model=self.model)
        checkpoint_prefix = os.path.join(checkpoint_dir, self.step)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, directory=checkpoint_prefix, max_to_keep=self.max_to_keep
        )
        self.checkpoint = (checkpoint, checkpoint_manager)

        predict_layers_path = os.path.join(checkpoint_dir, "predict_layers")
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
        value_to_load = kwargs.get("value_to_load", {})
        key_to_numpy = kwargs.get("key_to_numpy", {})
        pop_key = kwargs.get("pop_key", True)
        include_prediction_layer = kwargs.get("include_prediction_layer", True)

        if not bool(key_to_numpy):
            reader = tf.compat.v1.train.NewCheckpointReader(latest_checkpoint)
            var_to_shape_map = reader.get_variable_to_shape_map()
            value_to_load = {}
            key_to_numpy = {}
            for key in sorted(var_to_shape_map.items()):
                key_to_numpy.update({key[0]: reader.get_tensor(key[0])})

        key_maps1 = {
            "Conv1/kernel": "Conv/weights",
            "bn_Conv1/gamma": "Conv/BatchNorm/gamma",
            "bn_Conv1/beta": "Conv/BatchNorm/beta",
            "bn_Conv1/moving_mean": "Conv/BatchNorm/moving_mean",
            "bn_Conv1/moving_variance": "Conv/BatchNorm/moving_variance",
            "expanded_conv_depthwise/depthwise_kernel": "expanded_conv/depthwise/depthwise_weights",
            "expanded_conv_depthwise_BN/gamma": "expanded_conv/depthwise/BatchNorm/gamma",
            "expanded_conv_depthwise_BN/beta": "expanded_conv/depthwise/BatchNorm/beta",
            "expanded_conv_depthwise_BN/moving_mean": "expanded_conv/depthwise/BatchNorm/moving_mean",
            "expanded_conv_depthwise_BN/moving_variance": "expanded_conv/depthwise/BatchNorm/moving_variance",
            "expanded_conv_project/kernel": "expanded_conv/project/weights",
            "expanded_conv_project_BN/gamma": "expanded_conv/project/BatchNorm/gamma",
            "expanded_conv_project_BN/beta": "expanded_conv/project/BatchNorm/beta",
            "expanded_conv_project_BN/moving_mean": "expanded_conv/project/BatchNorm/moving_mean",
            "expanded_conv_project_BN/moving_variance": "expanded_conv/project/BatchNorm/moving_variance",
            "Conv_1/kernel": "Conv_1/weights",
            "Conv_1_bn/gamma": "Conv_1/BatchNorm/gamma",
            "Conv_1_bn/beta": "Conv_1/BatchNorm/beta",
            "Conv_1_bn/moving_mean": "Conv_1/BatchNorm/moving_mean",
            "Conv_1_bn/moving_variance": "Conv_1/BatchNorm/moving_variance",
        }
        key_maps2 = {
            "prediction_layer/prediction_layer/kernel": "Logits/Conv2d_1c_1x1/weights",
            "prediction_layer/prediction_layer/bias": "Logits/Conv2d_1c_1x1/biases",
        }
        key_maps3 = {
            "block_{}_expand/kernel": "expanded_conv_{}/expand/weights",
            "block_{}_expand_BN/gamma": "expanded_conv_{}/expand/BatchNorm/gamma",
            "block_{}_expand_BN/beta": "expanded_conv_{}/expand/BatchNorm/beta",
            "block_{}_expand_BN/moving_mean": "expanded_conv_{}/expand/BatchNorm/moving_mean",
            "block_{}_expand_BN/moving_variance": "expanded_conv_{}/expand/BatchNorm/moving_variance",
            "block_{}_depthwise/depthwise_kernel": "expanded_conv_{}/depthwise/depthwise_weights",
            "block_{}_depthwise_BN/gamma": "expanded_conv_{}/depthwise/BatchNorm/gamma",
            "block_{}_depthwise_BN/beta": "expanded_conv_{}/depthwise/BatchNorm/beta",
            "block_{}_depthwise_BN/moving_mean": "expanded_conv_{}/depthwise/BatchNorm/moving_mean",
            "block_{}_depthwise_BN/moving_variance": "expanded_conv_{}/depthwise/BatchNorm/moving_variance",
            "block_{}_project/kernel": "expanded_conv_{}/project/weights",
            "block_{}_project_BN/gamma": "expanded_conv_{}/project/BatchNorm/gamma",
            "block_{}_project_BN/beta": "expanded_conv_{}/project/BatchNorm/beta",
            "block_{}_project_BN/moving_mean": "expanded_conv_{}/project/BatchNorm/moving_mean",
            "block_{}_project_BN/moving_variance": "expanded_conv_{}/project/BatchNorm/moving_variance",
        }

        for key in key_maps1:
            _key = model_name + "/" + key_maps1[key]
            if _key in key_to_numpy:
                value = key_to_numpy[_key]
                value_to_load[target_model + key] = value
                if pop_key:
                    key_to_numpy.pop(_key)
            else:
                print("Can't find the key: {}".format(_key))

        if include_prediction_layer:
            for key in key_maps2:
                _key = model_name + "/" + key_maps2[key]
                if _key in key_to_numpy:
                    value = key_to_numpy[_key]
                    value_to_load[key] = value
                    if pop_key:
                        key_to_numpy.pop(_key)
                else:
                    print("Can't find the key: {}".format(_key))

        for i in range(1, 17):
            for key in key_maps3:
                k = model_name + "/" + key_maps3[key]
                _key_v = k.format(i)
                if _key_v in key_to_numpy:
                    value = key_to_numpy[_key_v]
                    value_to_load[target_model + key.format(i)] = value
                    if pop_key:
                        key_to_numpy.pop(_key_v)
                else:
                    print("Can't find the key: {}".format(_key_v))
        return value_to_load

    def load_from_pretrain1(
        self, latest_checkpoint, target_model="mobilenetv2", model_name="mobilenet_v2_140_stn_v15", **kwargs
    ):
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
        except:
            return 0
        else:
            return step

    def load_weights(self, checkpoint_dir):
        self.model.load_weights(checkpoint_dir)

    def load_model_variables(self):
        pass

    def restore_model_variables(self, checkpoint_dir=None, **kwargs):
        step = 1
        loaded_successfully = self.restore_model_from_latest_checkpoint_if_exist(checkpoint_dir=checkpoint_dir, **kwargs)
        if loaded_successfully:
            step = self.get_step_number_from_latest_checkpoint() + 1
        else:
            self.load_model_variables()

        self.config_layers()
        for var in self.model.trainable_variables:
            print("trainable variable: ", var.name)
        return step

    def config_layers(self):
        if self.step == TRAIN_STEP1:
            self.set_mobilenet_training_status(False)

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
            tf.io.gfile.mkdir(model_path)
        self.model.save(
            filepath=model_path,
            overwrite=overwrite,
            include_optimizer=include_optimizer,
            save_format=save_format,
            signatures=signatures,
            options=options,
        )


class PreprocessLayer(keras.layers.Layer):
    def __init__(self):
        super(PreprocessLayer, self).__init__()

    def call(self, inputs, **kwargs):
        training = kwargs.pop("training")
        if training:
            sel = tf.random.uniform([], maxval=4, dtype=tf.int32)
            inputs = tf.switch_case(
                sel,
                branch_fns={
                    0: lambda: tf.image.random_flip_left_right(inputs),
                    1: lambda: tf.image.random_flip_up_down(inputs),
                    2: lambda: tf.image.rot90(inputs),
                },
                default=lambda: inputs,
            )

            sel = tf.random.uniform([], maxval=5, dtype=tf.int32)
            inputs = tf.switch_case(
                sel,
                branch_fns={
                    0: lambda: tf.image.random_brightness(inputs, max_delta=0.5),
                    1: lambda: tf.image.random_saturation(inputs, lower=1, upper=5),
                    2: lambda: tf.image.random_contrast(inputs, lower=0.2, upper=0.5),
                    3: lambda: tf.image.random_hue(inputs, max_delta=0.2),
                },
                default=lambda: inputs,
            )
        return inputs


class PredictionLayer(keras.layers.Layer):
    def __init__(self, num_classes, name, activation=None, dropout_ratio=0.2):
        super(PredictionLayer, self).__init__()
        self.layer_name = name
        self.global_average_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout = keras.layers.Dropout(dropout_ratio)
        self.dense = keras.layers.Dense(num_classes, name="prediction_layer")
        self.prediction_fn = activations.get(activation)

    def call(self, inputs, **kwargs):
        training = kwargs.pop("training")
        inputs = self.global_average_pooling(inputs, training=training)
        if training:
            inputs = self.dropout(inputs, training=training)
        inputs = self.dense(inputs, training=training)
        inputs = self.prediction_fn(inputs)
        tf.summary.histogram(
            "prediction/weights/{}-kernel".format(self.layer_name),
            self.dense.weights[0],
            step=tf.compat.v1.train.get_global_step(),
        )
        tf.summary.histogram(
            "prediction/weights/{}-bias".format(self.layer_name),
            self.dense.weights[1],
            step=tf.compat.v1.train.get_global_step(),
        )
        tf.summary.histogram(
            "prediction/activation/{}".format(self.layer_name),
            inputs,
            step=tf.compat.v1.train.get_global_step(),
        )
        return inputs


def global_pool(shape, pool_op=keras.layers.AvgPool2D):
    pool_size = [shape[1], shape[2]]
    output = pool_op(pool_size=pool_size, strides=[1, 1], padding="valid")
    return output


def create_mobilenet_v2_14(num_classes, optimizer=None, loss_fn=None, training=False, **kwargs):
    step = kwargs.pop("step") if "step" in kwargs else TRAIN_TEMPLATE.format(1)

    inputs = keras.Input(shape=IMG_SHAPE_224)
    preprocess_layer = PreprocessLayer()
    mobilenet = create_mobilenet_v2(input_shape=IMG_SHAPE_224, alpha=1.4, include_top=False, weights=None)
    processed_inputs = preprocess_layer(inputs, training=training)
    mobilenet_logits = mobilenet(processed_inputs, training=training)

    prediction_layer = PredictionLayer(num_classes=num_classes, name="final-prediction")

    outputs = prediction_layer(mobilenet_logits, training=training)

    model = Orchids52Mobilenet140(
        inputs,
        outputs,
        optimizer=optimizer,
        loss_fn=loss_fn,
        mobilenet=mobilenet,
        predict_layers=[prediction_layer],
        training=training,
        step=step,
    )

    return model
