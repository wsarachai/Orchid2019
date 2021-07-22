from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from tensorflow.core.framework.types_pb2 import DT_DOUBLE
import tensorflow.keras as keras

from nets.const_vars import default_image_size, IMG_SHAPE_224
from nets.core_functions import load_orchids52_weight_from_old_checkpoint
from nets.layers import PredictionLayer, PreprocessLayer
from stn import pre_spatial_transformer_network
from tensorflow.python.keras import initializers
from tensorflow.python.keras import activations
from nets.mobilenet_v2 import create_mobilenet_v2
from nets.v1.mobilenet_v2_140 import Orchids52Mobilenet140
from utils.const import TRAIN_STEP1, TRAIN_STEP2, TRAIN_STEP3, TRAIN_STEP4, TRAIN_TEMPLATE
from utils.const import TRAIN_V2_STEP1, TRAIN_V2_STEP2


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

    def config_checkpoint(self, checkpoint_path):
        super(Orchids52Mobilenet140STN, self).config_checkpoint(checkpoint_path)
        if self.stn_denses and len(self.stn_denses) > 0:
            for i, stn_dense in enumerate(self.stn_denses):
                stn_dense_checkpoint = tf.train.Checkpoint(
                    step=tf.Variable(1), optimizer=self.optimizer, model=stn_dense
                )
                checkpoint_prefix = os.path.join(checkpoint_path, "stn_dense_layer_{}".format(i))
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

        key_to_numpy, var_maps = load_orchids52_weight_from_old_checkpoint(latest_checkpoint)

        if training_for_tf25:
            var_maps_ext = {
                "branch_block/prediction_layer/dense/kernel": "Logits/Conv2d_1c_1x1/weights",
                "branch_block/prediction_layer/dense/bias": "Logits/Conv2d_1c_1x1/biases",
                "branch_block/prediction_layer_1/dense_1/kernel": "Logits/Conv2d_1c_1x1/weights",
                "branch_block/prediction_layer_1/dense_1/bias": "Logits/Conv2d_1c_1x1/biases",
                "branch_block/prediction_layer_2/dense_2/kernel": "Logits/Conv2d_1c_1x1/weights",
                "branch_block/prediction_layer_2/dense_2/bias": "Logits/Conv2d_1c_1x1/biases",
            }
        else:
            var_maps_ext = {
                "branch_block/prediction_layer/dense/kernel": "Logits/Conv2d_1c_1x1-0/weights",
                "branch_block/prediction_layer/dense/bias": "Logits/Conv2d_1c_1x1-0/biases",
                "branch_block/prediction_layer_1/dense_1/kernel": "Logits/Conv2d_1c_1x1-1/weights",
                "branch_block/prediction_layer_1/dense_1/bias": "Logits/Conv2d_1c_1x1-1/biases",
                "branch_block/prediction_layer_2/dense_2/kernel": "Logits/Conv2d_1c_1x1-2/weights",
                "branch_block/prediction_layer_2/dense_2/bias": "Logits/Conv2d_1c_1x1-2/biases",
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

        local_var_loaded = super(Orchids52Mobilenet140STN, self).load_from_v1(
            latest_checkpoint,
            target_model + "_stn_base_1.40_224_",
            localization_params,
            key_to_numpy=key_to_numpy,
            include_prediction_layer=False,
            **kwargs
        )
        extract_var_loaded = super(Orchids52Mobilenet140STN, self).load_from_v1(
            latest_checkpoint,
            target_model + "_global_branch_1.40_224_",
            features_extraction,
            key_to_numpy=key_to_numpy,
            include_prediction_layer=False,
            **kwargs
        )
        extract_comm_var_loaded = super(Orchids52Mobilenet140STN, self).load_from_v1(
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
                if "RMSProp" not in key:
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
        if self.step == TRAIN_STEP1:
            self.set_mobilenet_training_status(False)
        elif self.step == TRAIN_STEP2:
            self.set_mobilenet_training_status(False)
            self.set_prediction_training_status(False)
        elif self.step == TRAIN_STEP3:
            self.set_mobilenet_training_status(False)
            self.set_prediction_training_status(False)
            for stn_dense in self.stn_denses:
                stn_dense.trainable = False
        elif self.step == TRAIN_V2_STEP1:
            self.set_mobilenet_training_status(False)
        elif self.step == TRAIN_V2_STEP2:
            self.set_mobilenet_training_status(True)

    def load_model_step2(self, **kwargs):
        self.load_model_step1()
        for checkpoint in self.stn_dense_checkpoints:
            stn_dense_checkpoint, stn_dense_checkpoint_manager = checkpoint
            if stn_dense_checkpoint_manager.latest_checkpoint:
                status = stn_dense_checkpoint.restore(stn_dense_checkpoint_manager.latest_checkpoint)
                status.assert_existing_objects_matched()

    def load_model_step3(self, **kwargs):
        self.load_model_step2()

    def load_model_step4(self, **kwargs):
        assert self.checkpoint_path is not None

        checkpoint, _ = self.checkpoint
        checkpoint_prefix = os.path.join(self.checkpoint_path, TRAIN_TEMPLATE.format(step=3))
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, directory=checkpoint_prefix, max_to_keep=self.max_to_keep
        )
        if checkpoint_manager.latest_checkpoint:
            status = checkpoint.restore(checkpoint_manager.latest_checkpoint)
            status.assert_existing_objects_matched()

    def load_model_v2_step2(self, **kwargs):
        pass

    def load_model_variables(self, **kwargs):
        if self.step == TRAIN_STEP1:
            self.load_model_step1(**kwargs)
        elif self.step == TRAIN_STEP2:
            self.load_model_step2(**kwargs)
        elif self.step == TRAIN_STEP3:
            self.load_model_step3(**kwargs)
        elif self.step == TRAIN_STEP4:
            self.load_model_step4(**kwargs)
        elif self.step == TRAIN_V2_STEP2:
            self.load_model_v2_step2(**kwargs)


class BranchBlock(keras.layers.Layer):
    def __init__(self, num_classes, batch_size, dropout=0.8, width=default_image_size, height=default_image_size):
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
            PredictionLayer(num_classes=num_classes, dropout_ratio=dropout),
            PredictionLayer(num_classes=num_classes, dropout_ratio=dropout),
            PredictionLayer(num_classes=num_classes, dropout_ratio=dropout),
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


def create_orchid_mobilenet_v2_15(num_classes, optimizer=None, loss_fn=None, training=False, dropout=0.8, **kwargs):
    stn_denses = None
    branches_block = None
    boundary_loss = None
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

    if step != TRAIN_STEP1:
        scales = [0.5, 0.3]
        fc_num = 3

        stn_dense1 = keras.Sequential(
            [
                keras.layers.Conv2D(128, [1, 1], activation="relu", name="stn_conv2d_1"),
                keras.layers.Flatten(),
                keras.layers.Dense(128, activation="tanh", name="stn_dense_128_1"),
                keras.layers.Dropout(rate=dropout),
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
                keras.layers.Dropout(rate=dropout),
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

        stn_out1 = keras.Sequential([stn_base_model, stn_dense1])
        stn_out2 = keras.Sequential([stn_base_model, stn_dense2])

        stn_logits1 = stn_out1(processed_inputs)
        stn_logits2 = stn_out2(processed_inputs)

        loc_output = keras.layers.Concatenate(axis=1)([stn_logits1, stn_logits2])

        # TODO: Change this function to keras layer
        stn_output, bound_err = pre_spatial_transformer_network(
            input_map=processed_inputs,
            theta=loc_output,
            batch_size=batch_size,
            width=default_image_size,
            height=default_image_size,
            scales=scales,
        )

        if training:
            with tf.name_scope("boundary_loss"):
                _len = bound_err.shape[0]
                bound_std = tf.constant(np.full(_len, 0.00, dtype=np.float32), name="bound_std")
                boundary_loss = keras.Model(inputs, keras.losses.MSE(bound_err, bound_std), name="mse")

        # with tf.name_scope('branches'):
        branches_block = BranchBlock(num_classes=num_classes, batch_size=batch_size)
        branches_prediction_models = branches_block.branches_prediction_models

        logits = branches_block(stn_output)

        # # with tf.name_scope('estimate_block'):
        if step == TRAIN_STEP2:
            outputs = tf.reduce_mean(logits, axis=0)
        else:
            estimate_block = EstimationBlock(num_classes=num_classes, batch_size=batch_size)
            outputs = estimate_block(logits)

    else:
        prediction_layer = PredictionLayer(
            num_classes=num_classes, dropout_ratio=dropout, activation="softmax", name=""
        )
        branches_prediction_models.append(prediction_layer)
        mobilenet_logits = stn_base_model(processed_inputs, training=training)
        outputs = prediction_layer(mobilenet_logits, training=training)

    if activation == "softmax":
        outputs = tf.keras.activations.softmax(outputs)

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
