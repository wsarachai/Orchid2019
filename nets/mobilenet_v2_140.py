from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

import nets
import lib_utils
import tensorflow as tf
import tensorflow.keras as keras
from nets import mobilenet_v2_orchids


def preprocess_input(image_data, central_fraction=0.875):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.central_crop(image, central_fraction=central_fraction)
    image = tf.image.resize(images=image, size=nets.mobilenet_v2.IMG_SIZE_224, method=tf.image.ResizeMethod.BILINEAR)
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
            prediction_layer_prefix = lib_utils.get_checkpoint_file(predict_layers_path, idx)
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

    def load_from_v1(self, latest_checkpoint, model_name="MobilenetV2"):
        reader = tf.compat.v1.train.NewCheckpointReader(latest_checkpoint)
        var_to_shape_map = reader.get_variable_to_shape_map()
        value_to_load = {}
        key_to_numpy = {}
        for key in sorted(var_to_shape_map.items()):
            key_to_numpy.update({key[0]: key[1]})

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
            "prediction_layer/prediction_layer/kernel": "Logits/Conv2d_1c_1x1/weights",
            "prediction_layer/prediction_layer/bias": "Logits/Conv2d_1c_1x1/biases",
        }
        key_maps2 = {
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
                value = reader.get_tensor(_key)
                value_to_load[key] = value
            else:
                print("Can't find the key: {}".format(_key))

        for i in range(1, 17):
            for key in key_maps2:
                _key = model_name + "/" + key_maps2[key]
                _key_v = _key.format(i)
                if _key_v in key_to_numpy:
                    value = reader.get_tensor(_key_v)
                    value_to_load[key.format(i)] = value
                else:
                    raise Exception("Can't find the key: {}".format(_key))
        return value_to_load

    def restore_model_from_latest_checkpoint_if_exist(self, **kwargs):
        result = False
        if self.checkpoint:
            checkpoint, checkpoint_manager = self.checkpoint
            if checkpoint_manager.latest_checkpoint:
                status = checkpoint.restore(checkpoint_manager.latest_checkpoint)
                status.assert_existing_objects_matched()
                result = True

        if not result:
            latest_checkpoint = kwargs.pop("checkpoint_path")
            if latest_checkpoint:
                var_loaded = self.load_from_v1(latest_checkpoint)
                var_loaded_fixed_name = {}
                for key in var_loaded:
                    var_loaded_fixed_name.update({key + ":0": var_loaded[key]})
                all_vars = self.model.weights
                for i, var in enumerate(all_vars):
                    print("Loading: ", var.name, var.shape)
                    if var.name in var_loaded_fixed_name:
                        saved_var = var_loaded_fixed_name[var.name]
                        if var.shape != saved_var.shape:
                            saved_var = np.squeeze(saved_var)
                            if var.shape != saved_var.shape:
                                raise Exception("Incompatible shapes")
                        tf.assert_equal(var.shape, saved_var.shape)
                        var.assign(saved_var)
                    else:
                        print("Can't find: {}".format(var.name))
                return True
            else:
                return False

    def get_step_number_from_latest_checkpoint(self):
        try:
            _, checkpoint_manager = self.checkpoint
            index = checkpoint_manager.latest_checkpoint.index("ckpt-")
            step = checkpoint_manager.latest_checkpoint[index:][5:]
            step = int(step)
        except:
            return 1
        else:
            return step

    def load_weights(self, checkpoint_path):
        self.model.load_weights(checkpoint_path)

    def restore_model_variables(self, load_from_checkpoint_first=True, checkpoint_path=None):
        step = 1
        loaded_successfully = False
        if load_from_checkpoint_first:
            loaded_successfully = self.restore_model_from_latest_checkpoint_if_exist(checkpoint_path=checkpoint_path)
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
            inputs = tf.switch_case(sel, branch_fns={
                0: lambda: tf.image.random_flip_left_right(inputs),
                1: lambda: tf.image.random_flip_up_down(inputs),
                2: lambda: tf.image.rot90(inputs),
            }, default=lambda: inputs)

            sel = tf.random.uniform([], maxval=5, dtype=tf.int32)
            inputs = tf.switch_case(sel, branch_fns={
                0: lambda: tf.image.random_brightness(inputs, max_delta=0.5),
                1: lambda: tf.image.random_saturation(inputs, lower=1, upper=5),
                2: lambda: tf.image.random_contrast(inputs, lower=0.2, upper=0.5),
                3: lambda: tf.image.random_hue(inputs, max_delta=0.2),
            }, default=lambda: inputs)

        inputs = tf.subtract(inputs, 0.5)
        inputs = tf.multiply(inputs, 2.0)
        return inputs


class PredictionLayer(keras.layers.Layer):
    def __init__(self, num_classes, shape, activation=None, dropout_ratio=0.2):
        super(PredictionLayer, self).__init__()
        self.global_average_pooling = global_pool(shape=shape)
        self.dropout = keras.layers.Dropout(dropout_ratio)
        # self.dense = keras.layers.Conv2D(
        #     num_classes,
        #     kernel_size=1,
        #     padding="same",
        #     use_bias=True,
        #     activation=activation,
        #     bias_initializer=tf.zeros_initializer(),
        #     name="dense-{}".format(num_classes),
        # )
        self.dense = keras.layers.Dense(num_classes, name="prediction_layer")
        self.prediction_fn = keras.layers.Softmax()

    def call(self, inputs, **kwargs):
        training = kwargs.pop("training")
        inputs = self.global_average_pooling(inputs, training=training)
        if training:
            inputs = self.dropout(inputs, training=training)
        inputs = self.dense(inputs, training=training)
        inputs = tf.squeeze(inputs, [1, 2])
        inputs = self.prediction_fn(inputs, training=training)
        return inputs


def global_pool(shape, pool_op=keras.layers.AvgPool2D):
    pool_size = [shape[1], shape[2]]
    output = pool_op(pool_size=pool_size, strides=[1, 1], padding="valid")
    return output


def create_mobilenet_v2_14(num_classes, optimizer, loss_fn, training=False, **kwargs):
    step = kwargs.pop("step") if "step" in kwargs else nets.utils.TRAIN_TEMPLATE.format(1)

    inputs = keras.Input(shape=nets.mobilenet_v2.IMG_SHAPE_224)
    preprocess_layer = PreprocessLayer()
    mobilenet = mobilenet_v2_orchids.create_mobilenet_v2(
        input_shape=nets.mobilenet_v2.IMG_SHAPE_224, alpha=1.4, include_top=False, weights=None
    )
    processed_inputs = preprocess_layer(inputs, training=training)
    mobilenet_logits = mobilenet(processed_inputs, training=training)

    prediction_layer = PredictionLayer(num_classes=num_classes, shape=mobilenet_logits.get_shape().as_list())

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
