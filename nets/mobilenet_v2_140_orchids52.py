from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import nets

from absl import logging
from stn import SpatialTransformerNetwork
from tensorflow.python.keras import initializers
from tensorflow.python.keras import activations
from nets.mobilenet_v2 import IMG_SHAPE_224
from nets.mobilenet_v2 import default_image_size
from nets.mobilenet_v2 import create_mobilenet_v2
from nets.mobilenet_v2_140 import PreprocessLayer
from nets.mobilenet_v2_140 import PredictionLayer
from nets.mobilenet_v2_140 import Orchids52Mobilenet140
from utils.const import TRAIN_STEP1, TRAIN_STEP2, TRAIN_STEP3, TRAIN_STEP4
from utils.const import TRAIN_V2_STEP1, TRAIN_V2_STEP2
from utils.const import TRAIN_TEMPLATE


class Orchids52Mobilenet140STN(Orchids52Mobilenet140):
    def __init__(
        self,
        inputs,
        outputs,
        optimizer,
        loss_fn,
        base_model,
        stn_denses,
        estimate_block,
        predict_models,
        branch_model,
        boundary_loss,
        training,
        step,
    ):
        super(Orchids52Mobilenet140STN, self).__init__(
            inputs, outputs, optimizer, loss_fn, base_model, predict_models, training, step
        )
        self.stn_denses = stn_denses
        self.branch_model = branch_model
        self.estimate_block = estimate_block
        self.boundary_loss = boundary_loss
        self.stn_dense_checkpoints = []

    def config_checkpoint(self, checkpoint_dir):
        super(Orchids52Mobilenet140STN, self).config_checkpoint(checkpoint_dir)
        if self.stn_denses and len(self.stn_denses) > 0:
            for i, stn_dense in enumerate(self.stn_denses):
                stn_dense_checkpoint = tf.train.Checkpoint(
                    step=tf.Variable(1), optimizer=self.optimizer, model=stn_dense
                )
                checkpoint_prefix = os.path.join(checkpoint_dir, "stn_dense_layer_{}".format(i))
                stn_dense_checkpoint_manager = tf.train.CheckpointManager(
                    stn_dense_checkpoint, directory=checkpoint_prefix, max_to_keep=self.max_to_keep
                )
                self.stn_dense_checkpoints.append((stn_dense_checkpoint, stn_dense_checkpoint_manager))

    def load_from_v1(
        self, latest_checkpoint, target_model="mobilenetv2", model_name="mobilenet_v2_140_stn_v15", **kwargs
    ):
        training_for_tf25 = kwargs.get("training_for_tf25", False)
        pop_key = kwargs.get("pop_key", True)
        value_to_load = {}
        reader = tf.compat.v1.train.NewCheckpointReader(latest_checkpoint)
        var_to_shape_map = reader.get_variable_to_shape_map()
        key_to_numpy = {}
        for key in sorted(var_to_shape_map.items()):
            key_to_numpy.update({key[0]: reader.get_tensor(key[0])})

        var_maps = {
            "stn_conv2d_1/kernel": "dense-1/conv2d_resize_128/weights",
            "stn_conv2d_1/bias": "dense-1/conv2d_resize_128/biases",
            "stn_dense_128_1/kernel": "dense-1/fc_128/weights",
            "stn_dense_128_1/bias": "dense-1/fc_128/biases",
            "stn_dense_3_1/kernel": "dense-1/fc_final-1/weights",
            "stn_conv2d_2/kernel": "dense-2/conv2d_resize_128/weights",
            "stn_conv2d_2/bias": "dense-2/conv2d_resize_128/biases",
            "stn_dense_128_2/kernel": "dense-2/fc_128/weights",
            "stn_dense_128_2/bias": "dense-2/fc_128/biases",
            "stn_dense_3_2/kernel": "dense-2/fc_final-2/weights",
            "estimation_block/fully_connected_layer/kernel": "Estimation/fully_connected_logits/weights",
            "estimation_block/batch_normalization/gamma": "Estimation/fully_connected_logits/BatchNorm/gamma",
            "estimation_block/batch_normalization/beta": "Estimation/fully_connected_logits/BatchNorm/beta",
            "estimation_block/batch_normalization/moving_mean": "Estimation/fully_connected_logits/BatchNorm/moving_mean",
            "estimation_block/batch_normalization/moving_variance": "Estimation/fully_connected_logits/BatchNorm/moving_variance",
        }

        if training_for_tf25:
            var_maps_ext = {
                "branch_block/prediction_layer/prediction_layer/kernel": "Logits/Conv2d_1c_1x1/weights",
                "branch_block/prediction_layer/prediction_layer/bias": "Logits/Conv2d_1c_1x1/biases",
                "branch_block/prediction_layer_1/prediction_layer/kernel": "Logits/Conv2d_1c_1x1/weights",
                "branch_block/prediction_layer_1/prediction_layer/bias": "Logits/Conv2d_1c_1x1/biases",
                "branch_block/prediction_layer_2/prediction_layer/kernel": "Logits/Conv2d_1c_1x1/weights",
                "branch_block/prediction_layer_2/prediction_layer/bias": "Logits/Conv2d_1c_1x1/biases",
            }
        else:
            var_maps_ext = {
                "branch_block/prediction_layer/prediction_layer/kernel": "Logits/Conv2d_1c_1x1-0/weights",
                "branch_block/prediction_layer/prediction_layer/bias": "Logits/Conv2d_1c_1x1-0/biases",
                "branch_block/prediction_layer_1/prediction_layer/kernel": "Logits/Conv2d_1c_1x1-1/weights",
                "branch_block/prediction_layer_1/prediction_layer/bias": "Logits/Conv2d_1c_1x1-1/biases",
                "branch_block/prediction_layer_2/prediction_layer/kernel": "Logits/Conv2d_1c_1x1-2/weights",
                "branch_block/prediction_layer_2/prediction_layer/bias": "Logits/Conv2d_1c_1x1-2/biases",
            }

        var_maps.update(var_maps_ext)

        if training_for_tf25:
            localization_params = "MobilenetV2"
            features_extraction = "MobilenetV2"
            features_extraction_common = "MobilenetV2"
        else:
            localization_params = model_name + "/localization_params/MobilenetV2"
            features_extraction = model_name + "/features-extraction/MobilenetV2"
            features_extraction_common = model_name + "/features-extraction-common/MobilenetV2"

        local_var_loaded = super().load_from_v1(
            latest_checkpoint,
            target_model + "_stn_base_1.40_224_",
            localization_params,
            key_to_numpy=key_to_numpy,
            include_prediction_layer=False,
            **kwargs
        )
        extract_var_loaded = super().load_from_v1(
            latest_checkpoint,
            target_model + "_global_branch_1.40_224_",
            features_extraction,
            key_to_numpy=key_to_numpy,
            include_prediction_layer=False,
            **kwargs
        )
        extract_comm_var_loaded = super().load_from_v1(
            latest_checkpoint,
            target_model + "_shared_branch_1.40_224_",
            features_extraction_common,
            key_to_numpy=key_to_numpy,
            include_prediction_layer=False,
            **kwargs
        )

        for var_name in var_maps:
            if training_for_tf25:
                _key = "MobilenetV2/" + var_maps[var_name]
            else:
                _key = model_name + "/" + var_maps[var_name]
            if _key in key_to_numpy:
                value = key_to_numpy[_key]
                value_to_load[var_name] = value
                if pop_key:
                    key_to_numpy.pop(_key)
            else:
                print("Can't find the key: {}".format(_key))

        for key in local_var_loaded:
            value_to_load[key] = local_var_loaded[key]
        for key in extract_var_loaded:
            value_to_load[key] = extract_var_loaded[key]
        for key in extract_comm_var_loaded:
            value_to_load[key] = extract_comm_var_loaded[key]

        if pop_key:
            for key in key_to_numpy:
                print("{} was not loaded".format(key))

        return value_to_load

    def save_model_variables(self):
        super(Orchids52Mobilenet140STN, self).save_model_variables()
        if len(self.stn_dense_checkpoints) > 0:
            for checkpoint in self.stn_dense_checkpoints:
                _, stn_dense_checkpoint_manager = checkpoint
                stn_dense_checkpoint_manager.save()

    def set_mobilenet_training_status(self, trainable):
        super(Orchids52Mobilenet140STN, self).set_mobilenet_training_status(trainable)
        if self.branch_model:
            self.branch_model.set_trainable_for_global_branch(trainable)
            self.branch_model.set_trainable_for_share_branch(trainable)

    def config_layers(self):
        training_step = TRAIN_TEMPLATE.format(self.step)
        if training_step == TRAIN_STEP1:
            self.set_mobilenet_training_status(False)
        elif training_step == TRAIN_STEP2:
            self.set_mobilenet_training_status(False)
            self.set_prediction_training_status(False)
            self.stn_denses[0].trainable = False
        elif training_step == TRAIN_STEP3:
            self.set_mobilenet_training_status(False)
            self.set_prediction_training_status(False)
            self.stn_denses[1].trainable = False
        elif training_step == TRAIN_STEP4:
            self.set_mobilenet_training_status(False)
            self.set_prediction_training_status(False)
            for stn_dense in self.stn_denses:
                stn_dense.trainable = False
        elif training_step == TRAIN_V2_STEP1:
            self.set_mobilenet_training_status(False)
        elif training_step == TRAIN_V2_STEP2:
            self.set_mobilenet_training_status(True)

    def load_model_step1(self, checkpoint_dir):
        if self.checkpoint:
            checkpoint, checkpoint_manager = self.checkpoint
            if checkpoint_dir:
                latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
                checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, model=self.mobilenet)
                status = checkpoint.restore(latest_checkpoint)
                status.expect_partial()

                checkpoint = tf.train.Checkpoint(
                    step=tf.Variable(1), optimizer=self.optimizer, model=self.branch_model.global_branch_model
                )
                status = checkpoint.restore(latest_checkpoint)
                status.expect_partial()

                checkpoint = tf.train.Checkpoint(
                    step=tf.Variable(1), optimizer=self.optimizer, model=self.branch_model.shared_branch_model
                )
                status = checkpoint.restore(latest_checkpoint)
                status.expect_partial()

            latest_checkpoint = None
            for checkpoint, checkpoint_manager in self.prediction_layer_checkpoints:
                if checkpoint_manager.latest_checkpoint:
                    latest_checkpoint = checkpoint_manager.latest_checkpoint
                if latest_checkpoint:
                    status = checkpoint.restore(latest_checkpoint)
                    status.assert_existing_objects_matched()

    def load_model_step2(self, checkpoint_dir):
        self.load_model_step1(os.path.join(checkpoint_dir, TRAIN_TEMPLATE.format(self.step - 1)))
        for checkpoint in self.stn_dense_checkpoints:
            stn_dense_checkpoint, stn_dense_checkpoint_manager = checkpoint
            if stn_dense_checkpoint_manager.latest_checkpoint:
                status = stn_dense_checkpoint.restore(stn_dense_checkpoint_manager.latest_checkpoint)
                status.assert_existing_objects_matched()

    def load_model_step3(self):
        self.load_model_step2()

    def load_model_step4(self):
        assert self.checkpoint_dir is not None

        checkpoint, _ = self.checkpoint
        checkpoint_prefix = os.path.join(self.checkpoint_dir, TRAIN_TEMPLATE.format(3))
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, directory=checkpoint_prefix, max_to_keep=self.max_to_keep
        )
        if checkpoint_manager.latest_checkpoint:
            status = checkpoint.restore(checkpoint_manager.latest_checkpoint)
            status.assert_existing_objects_matched()

    def load_model_v2_step2(self):
        pass

    def load_model_variables(self, checkpoint_dir):
        training_step = TRAIN_TEMPLATE.format(self.step)
        if training_step == TRAIN_STEP1:
            self.load_model_step1(TRAIN_TEMPLATE.format(self.step))
        elif training_step == TRAIN_STEP2:
            self.load_model_step2(checkpoint_dir)
        elif training_step == TRAIN_STEP3:
            self.load_model_step3()
        elif training_step == TRAIN_STEP4:
            self.load_model_step4()
        elif training_step == TRAIN_V2_STEP2:
            self.load_model_v2_step2()

    def restore_model_from_latest_checkpoint_if_exist(self, **kwargs):
        result = False
        load_from_old_format = False
        show_model_weights = kwargs.get("show_model_weights", False)
        step = kwargs.get("training_step", 0)
        training_step = TRAIN_TEMPLATE.format(step)

        check_missing_weights = False if training_step == TRAIN_STEP4 else True

        if self.checkpoint:
            checkpoint, checkpoint_manager = self.checkpoint
            if checkpoint_manager.latest_checkpoint:
                try:
                    status = checkpoint.restore(checkpoint_manager.latest_checkpoint)
                    if check_missing_weights:
                        status.assert_existing_objects_matched()
                    else:
                        status.expect_partial()
                    result = True
                except:
                    pass
            else:
                load_from_old_format = True

        if training_step == TRAIN_STEP1 and load_from_old_format:
            var_loaded = None
            latest_checkpoint = kwargs.pop("checkpoint_dir")
            if latest_checkpoint:
                if self.training:
                    var_loaded = Orchids52Mobilenet140.load_from_v1(
                        self,
                        latest_checkpoint=latest_checkpoint,
                        target_model="mobilenetv2_stn_base_1.40_224_",
                        model_name="MobilenetV2",
                        include_prediction_layer=True,
                    )
                else:
                    var_loaded = self.load_from_v1(latest_checkpoint, **kwargs)

                if var_loaded:
                    var_loaded_fixed_name = {}
                    for key in var_loaded:
                        var_loaded_fixed_name.update({key + ":0": var_loaded[key]})

                    all_vars = self.model.weights

                    all_maps = {}
                    for _, var in enumerate(all_vars):
                        all_maps.update({var.name: var})

                    for _, var in enumerate(all_vars):
                        if var.name in var_loaded_fixed_name:
                            saved_var = var_loaded_fixed_name[var.name]
                            if var.shape != saved_var.shape:
                                saved_var = np.squeeze(saved_var)
                                if var.shape != saved_var.shape:
                                    raise Exception("Incompatible shapes")
                            var.assign(saved_var)
                            all_maps.pop(var.name)
                            if show_model_weights:
                                flat_var = np.reshape(saved_var, (-1))
                                print("Loading: {} -> {}".format(var.name, flat_var[:4]))
                        else:
                            print("Can't find: {}".format(var.name))

                    for key in all_maps:
                        var = all_maps[key]
                        print("Variable {} {} was not init..".format(var.name, var.shape))

                    result = True
            else:
                result = False
        return result


class BranchBlock(keras.layers.Layer):
    def __init__(self, num_classes, batch_size, width=default_image_size, height=default_image_size):
        super(BranchBlock, self).__init__()
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.global_branch_model = create_mobilenet_v2(
            input_shape=IMG_SHAPE_224, alpha=1.4, include_top=False, weights="imagenet", sub_name="global_branch"
        )
        self.shared_branch_model = create_mobilenet_v2(
            input_shape=IMG_SHAPE_224, alpha=1.4, include_top=False, weights="imagenet", sub_name="shared_branch"
        )
        self.branches_prediction_models = [
            PredictionLayer(num_classes=num_classes, name="BranchBlock-global"),
            PredictionLayer(num_classes=num_classes, name="BranchBlock-1"),
            PredictionLayer(num_classes=num_classes, name="BranchBlock-2"),
        ]

    def call(self, inputs, **kwargs):
        inp1 = tf.reshape(
            tf.slice(inputs, [0, 0, 0, 0, 0], [1, self.batch_size, self.width, self.height, 3]),
            [self.batch_size, self.width, self.height, 3],
        )
        inp2 = tf.reshape(
            tf.slice(inputs, [1, 0, 0, 0, 0], [1, self.batch_size, self.width, self.height, 3]),
            [self.batch_size, self.width, self.height, 3],
        )
        inp3 = tf.reshape(
            tf.slice(inputs, [2, 0, 0, 0, 0], [1, self.batch_size, self.width, self.height, 3]),
            [self.batch_size, self.width, self.height, 3],
        )

        logits = tf.stack(
            [
                self.sub_process(1, inp1, self.branches_prediction_models[0]),
                self.sub_process(2, inp2, self.branches_prediction_models[1]),
                self.sub_process(2, inp3, self.branches_prediction_models[2]),
            ],
            axis=0,
        )

        return logits

    def sub_process(self, branch, inp, prediction):
        if branch == 1:
            x = self.global_branch_model(inp, training=False)
            x = prediction(x, training=False)
        else:
            x = self.shared_branch_model(inp, training=False)
            x = prediction(x, training=False)
        return x

    def set_trainable_for_global_branch(self, trainable):
        self.global_branch_model.trainable = trainable

    def set_trainable_for_share_branch(self, trainable):
        self.shared_branch_model.trainable = trainable


class EstimationBlock(keras.layers.Layer):
    def __init__(self, num_classes, batch_size):
        super(EstimationBlock, self).__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.dense = FullyConnectedLayer(
            self.num_classes, kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.5), activation=None
        )
        self.batch_norm = tf.keras.layers.BatchNormalization(
            beta_initializer="zeros",
            gamma_initializer="ones",
            moving_mean_initializer="zeros",
            moving_variance_initializer="ones",
        )

    def call(self, inputs, **kwargs):
        inp1 = tf.reshape(
            tf.slice(inputs, [0, 0, 0], [1, self.batch_size, self.num_classes]), [self.batch_size, self.num_classes]
        )
        inp2 = tf.reshape(
            tf.slice(inputs, [1, 0, 0], [1, self.batch_size, self.num_classes]), [self.batch_size, self.num_classes]
        )
        inp3 = tf.reshape(
            tf.slice(inputs, [2, 0, 0], [1, self.batch_size, self.num_classes]), [self.batch_size, self.num_classes]
        )

        main_net = c_t = inp1

        input_and_hstate_concatenated = tf.concat(values=[c_t, inp2], axis=1)
        c_t = self.dense(input_and_hstate_concatenated)
        c_t = self.batch_norm(c_t)
        main_net = main_net + c_t

        input_and_hstate_concatenated = tf.concat(values=[c_t, inp3], axis=1)
        c_t = self.dense(input_and_hstate_concatenated)
        c_t = self.batch_norm(c_t)
        main_net = main_net + c_t

        return main_net


class FullyConnectedLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs, kernel_initializer, activation, normalizer_fn=None, **kwargs):
        super(FullyConnectedLayer, self).__init__(**kwargs)
        self.kernel = None
        self.num_outputs = num_outputs
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.normalizer_fn = normalizer_fn
        self.activation = activations.get(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            "kernel", shape=[int(input_shape[-1]), self.num_outputs], initializer=self.kernel_initializer
        )

    def call(self, inputs, **kwargs):
        x = tf.matmul(inputs, self.kernel)
        if self.normalizer_fn is not None:
            x = self.normalizer_fn(x)
        return self.activation(x)


class PrintingNode(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PrintingNode, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return tf.compat.v1.Print(inputs, [inputs])


def create_orchid_mobilenet_v2_15(
    num_classes, optimizer=None, loss_fn=None, training=False, drop_out_prop=0.8, **kwargs
):
    stn_denses = None
    boundary_loss = None
    branches_block = None
    estimate_block = None
    branches_prediction_models = []
    step = kwargs.pop("step") if "step" in kwargs else ""
    batch_size = kwargs.pop("batch_size") if "batch_size" in kwargs else 1
    activation = kwargs.pop("activation") if "activation" in kwargs else None

    inputs = keras.Input(shape=IMG_SHAPE_224)
    preprocess_layer = PreprocessLayer()
    stn_base_model = create_mobilenet_v2(
        input_shape=IMG_SHAPE_224, alpha=1.4, include_top=False, weights="imagenet", sub_name="stn_base"
    )

    processed_inputs = preprocess_layer(inputs, training=training)

    train_step = TRAIN_TEMPLATE.format(step)
    if train_step != TRAIN_STEP1:
        scales = [0.5, 0.3]
        fc_num = 2

        if train_step == TRAIN_STEP2:
            scales = [1.0, 0.3]

        if train_step == TRAIN_STEP3:
            scales = [0.5, 1.0]

        stn_dense1 = keras.Sequential(
            [
                keras.layers.Conv2D(128, [1, 1], activation="relu", name="stn_conv2d_1"),
                keras.layers.Flatten(),
                keras.layers.Dense(128, activation="tanh", name="stn_dense_128_1"),
                keras.layers.Dropout(rate=drop_out_prop),
                FullyConnectedLayer(
                    fc_num,
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.4),
                    normalizer_fn=tf.math.l2_normalize,
                    activation="tanh",
                    name="stn_dense_3_1",
                ),
            ],
            name="stn_dense1",
        )

        stn_dense2 = keras.Sequential(
            [
                keras.layers.Conv2D(128, [1, 1], activation="relu", name="stn_conv2d_2"),
                keras.layers.Flatten(),
                keras.layers.Dense(128, activation="tanh", name="stn_dense_128_2"),
                keras.layers.Dropout(rate=drop_out_prop),
                FullyConnectedLayer(
                    fc_num,
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.4),
                    normalizer_fn=tf.math.l2_normalize,
                    activation="tanh",
                    name="stn_dense_3_2",
                ),
            ],
            name="stn_dense2",
        )

        stn_denses = [stn_dense1, stn_dense2]

        stn_logits = stn_base_model(processed_inputs)
        stn_logits1 = stn_dense1(stn_logits)
        stn_logits2 = stn_dense2(stn_logits)

        stn_layer = SpatialTransformerNetwork(
            batch_size=batch_size, width=default_image_size, height=default_image_size, scales=scales
        )

        stn_output, bound_err = stn_layer(processed_inputs, thetas=[stn_logits1, stn_logits2])

        if training:
            bound_std = tf.constant(np.full(bound_err.shape, 0.00, dtype=np.float32), name="bound_std_zero")
            boundary_loss = keras.Model(inputs, keras.losses.MSE(bound_err, bound_std), name="mse")

        branches_block = BranchBlock(num_classes=num_classes, batch_size=batch_size)
        branches_prediction_models = branches_block.branches_prediction_models

        logits = branches_block(stn_output)

        if train_step == TRAIN_STEP2 or train_step == TRAIN_STEP3:
            outputs = tf.reduce_mean(logits, axis=0)
        else:
            estimate_block = EstimationBlock(num_classes=num_classes, batch_size=batch_size)
            outputs = estimate_block(logits)

        if activation == "softmax":
            outputs = tf.keras.activations.softmax(outputs)

    else:
        prediction_layer = PredictionLayer(num_classes=num_classes, activation="softmax", name=step)
        branches_prediction_models.append(prediction_layer)
        mobilenet_logits = stn_base_model(processed_inputs, training=training)
        outputs = prediction_layer(mobilenet_logits, training=training)

    model = Orchids52Mobilenet140STN(
        inputs,
        outputs,
        optimizer=optimizer,
        loss_fn=loss_fn,
        base_model=stn_base_model,
        stn_denses=stn_denses,
        estimate_block=estimate_block,
        predict_models=branches_prediction_models,
        branch_model=branches_block,
        boundary_loss=boundary_loss,
        training=training,
        step=step,
    )
    return model
