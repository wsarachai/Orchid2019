from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import h5py
import numpy as np
from six import b
import tensorflow as tf
import tensorflow.keras as keras

from absl import logging
from nets.const_vars import IMG_SHAPE_224, default_image_size
from nets.core_functions import load_orchids52_weight_from_old_checkpoint, load_weight
from stn import SpatialTransformerNetwork
from nets.mobilenet_v2 import create_mobilenet_v2
from nets.mobilenet_v2_140 import Orchids52Mobilenet140
from utils.const import TRAIN_STEP1, TRAIN_STEP2, TRAIN_STEP3, TRAIN_STEP4, TRAIN_STEP5
from utils.const import TRAIN_V2_STEP2
from utils.const import TRAIN_TEMPLATE
from nets.layers import Conv2DWrapper, DenseWrapper, PreprocessLayer, PredictionLayer
from nets.layers import FullyConnectedLayer
from nets.layers import EstimationBlock


def get_dataset_keys(f):
    keys = []
    f.visit(lambda key: keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys


def load_model_from_hdf5(filepath, model, optimizer, loaded_vars, to_name=None, from_name=None, **kwargs):
    verbose = kwargs.get("verbose", 0)
    file_loaded = None
    try:
        filepath = filepath + ".h5"
        file_loaded = h5py.File(filepath, mode="r")
        g = file_loaded["model_weights"]
        if verbose == 1:
            keys = get_dataset_keys(g)
            for v in keys:
                print(v)
        for var in model.weights:
            if var.ref() not in loaded_vars:
                if verbose == 1:
                    print(var.name)
                var_name = var.name

                if var_name not in g:
                    if to_name is not None and from_name is not None:
                        var_name = var_name.replace(from_name, to_name)
                if var_name in g:
                    weight = np.asarray(g[var_name])
                    if var.shape != weight.shape:
                        raise Exception("Incompatible shapes")
                    var.assign(weight)
                    loaded_vars[var.ref()] = var
                    if verbose == 1:
                        logging.info("%s is loaded...", var.name)
                else:
                    logging.warning("Variable [%s] is not loaded..", var_name)
        g = file_loaded["model_optimizer"]
        if verbose == 1:
            keys = get_dataset_keys(g)
            for v in keys:
                print(v)

        if optimizer:
            for var in optimizer.weights:
                if var.ref() not in loaded_vars:
                    if verbose == 1:
                        print(var.name)
                    var_name = var.name

                    if var_name not in g:
                        if to_name is not None and from_name is not None:
                            var_name = var_name.replace(from_name, to_name)
                    if var_name in g:
                        weight = np.asarray(g[var_name])
                        if var.shape != weight.shape:
                            raise Exception("Incompatible shapes")
                        var.assign(weight)
                        loaded_vars[var.ref()] = var
                        logging.info("%s is loaded...", var.name)

    except TypeError as e:
        logging.warning("Invalid filename [%s] not found with error: %s", filepath, e)
    finally:
        if file_loaded:
            file_loaded.close()


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
        self.loaded_vars = {}

    def config_checkpoint(self, checkpoint_dir):
        super(Orchids52Mobilenet140STN, self).config_checkpoint(checkpoint_dir)

    def load_from_v1(
        self, latest_checkpoint, target_model="mobilenetv2", model_name="mobilenet_v2_140_stn_v15", **kwargs
    ):
        training_for_tf25 = kwargs.get("training_for_tf25", False)
        pop_key = kwargs.get("pop_key", True)
        value_to_load = {}

        key_to_numpy, var_maps = load_orchids52_weight_from_old_checkpoint(latest_checkpoint)

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

        local_var_loaded = super(Orchids52Mobilenet140STN, self).load_from_v1(
            latest_checkpoint,
            target_model + "_stn_base_1.40_224_",
            localization_params,
            key_to_numpy=key_to_numpy,
            include_prediction_layer=False,
            **kwargs,
        )
        extract_var_loaded = super(Orchids52Mobilenet140STN, self).load_from_v1(
            latest_checkpoint,
            target_model + "_global_branch_1.40_224_",
            features_extraction,
            key_to_numpy=key_to_numpy,
            include_prediction_layer=False,
            **kwargs,
        )
        extract_comm_var_loaded = super(Orchids52Mobilenet140STN, self).load_from_v1(
            latest_checkpoint,
            target_model + "_shared_branch_1.40_224_",
            features_extraction_common,
            key_to_numpy=key_to_numpy,
            include_prediction_layer=False,
            **kwargs,
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

        # if self.stn_denses and len(self.stn_denses) > 0:
        #     for i, stn_dense in enumerate(self.stn_denses):
        #         checkpoint_prefix = os.path.join(self.checkpoint_dir, "stn_dense_layer")
        #         if not tf.io.gfile.exists(checkpoint_prefix):
        #             tf.io.gfile.makedirs(checkpoint_prefix)
        #         save_h5_weights(checkpoint_prefix + "/stn_dense_{}".format(i), stn_dense.weights)

    def config_layers(self, fine_tune, fine_tune_at=100, **kwargs):
        if self.step == 1:
            self.set_mobilenet_training_status(fine_tune, fine_tune_at=fine_tune_at)
        elif self.step == 2:
            self.set_mobilenet_training_status(False, fine_tune_at=fine_tune_at)
            if self.branch_model:
                for m in self.branch_model:
                    if m:
                        m.trainable = False
            self.set_prediction_training_status(False, **kwargs)
            if self.stn_denses[0]:
                self.stn_denses[0].trainable = False

        elif self.step == 3:
            self.set_mobilenet_training_status(False, fine_tune_at=fine_tune_at)
            if self.branch_model:
                for m in self.branch_model:
                    if m:
                        m.trainable = False
            self.set_prediction_training_status(False, **kwargs)
            if self.stn_denses[1]:
                self.stn_denses[1].trainable = False

        elif self.step == 4:
            self.set_mobilenet_training_status(fine_tune, fine_tune_at=fine_tune_at)
            if self.branch_model:
                for m in self.branch_model:
                    if m:
                        m.trainable = False
            self.set_prediction_training_status(False, **kwargs)

        elif self.step == 5:
            self.set_mobilenet_training_status(fine_tune, fine_tune_at=fine_tune_at)
            if self.branch_model:
                for m in self.branch_model:
                    if fine_tune:
                        if m:
                            m.trainable = True

                        if fine_tune_at:
                            # Freeze all the layers before the `fine_tune_at` layer
                            for layer in m.layers[:fine_tune_at]:
                                layer.trainable = False
                    else:
                        m.trainable = False

            self.set_prediction_training_status(False, **kwargs)
            self.estimate_block.trainable = True
            for stn_dense in self.stn_denses:
                stn_dense.trainable = False
        elif self.step == TRAIN_V2_STEP2:
            self.set_mobilenet_training_status(True, **kwargs)

    def load_model_step1(self, **kwargs):
        checkpoint_dir = os.path.join(self.checkpoint_dir, TRAIN_TEMPLATE.format(1))
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)

        load_model_from_hdf5(
            filepath=latest_checkpoint,
            model=self.mobilenet,
            optimizer=self.optimizer,
            loaded_vars=self.loaded_vars,
            **kwargs,
        )

        if self.step > 1:
            for idx, predict_layer in enumerate(self.predict_layers):
                if predict_layer:
                    from_name = (
                        "branch_block/prediction_layer/dense" if idx == 0 else "branch{}_prediction/dense".format(idx)
                    )

                    load_model_from_hdf5(
                        filepath=latest_checkpoint,
                        model=predict_layer,
                        optimizer=self.optimizer,
                        loaded_vars=self.loaded_vars,
                        from_name=from_name,
                        to_name="prediction_layer/dense",
                    )

            # _prediction_layer_prefix = ""
            # predict_layers_path = os.path.join(self.checkpoint_dir, "predict_layers")
            # for idx, predict_layer in enumerate(self.predict_layers):
            #     prediction_layer_prefix = get_checkpoint_file(predict_layers_path, idx)
            #     if not tf.io.gfile.exists(prediction_layer_prefix + ".h5"):
            #         prediction_layer_prefix = _prediction_layer_prefix
            #         if not tf.io.gfile.exists(prediction_layer_prefix + ".h5"):
            #             prediction_layer_prefix = latest_checkpoint
            #
            #     from_name = (
            #         "branch_block/prediction_layer/dense"
            #         if idx == 0
            #         else "branch_block/prediction_layer_{}/dense_{}".format(idx, idx)
            #     )
            #
            #     load_model_from_hdf5(
            #         filepath=prediction_layer_prefix,
            #         model=predict_layer,
            #         optimizer=self.optimizer,
            #         from_name=from_name,
            #         to_name="prediction_layer/dense",
            #     )
            #     _prediction_layer_prefix = prediction_layer_prefix
        else:
            for predict_layer in self.predict_layers:
                load_model_from_hdf5(
                    filepath=latest_checkpoint,
                    model=predict_layer,
                    optimizer=self.optimizer,
                    loaded_vars=self.loaded_vars,
                    **kwargs,
                )

    def load_model_step2(self, **kwargs):
        self.load_model_step1()

        checkpoint_dir = os.path.join(self.checkpoint_dir, TRAIN_TEMPLATE.format(1))
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)

        if self.branch_model[0]:
            load_model_from_hdf5(
                filepath=latest_checkpoint,
                model=self.branch_model[0],
                optimizer=self.optimizer,
                loaded_vars=self.loaded_vars,
                to_name="mobilenetv2_stn_base_1.40_224_",
                from_name="mobilenetv2_global_branch_1.40_224_",
            )
        if self.branch_model[1]:
            load_model_from_hdf5(
                filepath=latest_checkpoint,
                model=self.branch_model[1],
                optimizer=self.optimizer,
                loaded_vars=self.loaded_vars,
                to_name="mobilenetv2_stn_base_1.40_224_",
                from_name="mobilenetv2_shared_branch_1.40_224_",
            )

        if self.stn_denses and len(self.stn_denses) > 0:
            for i, stn_dense in enumerate(self.stn_denses):
                if stn_dense:
                    checkpoint_prefix = os.path.join(self.checkpoint_dir, "stn_dense_layer")
                    if tf.io.gfile.exists(checkpoint_prefix):
                        load_model_from_hdf5(
                            filepath=checkpoint_prefix + "/stn_dense_{}".format(i),
                            model=stn_dense,
                            optimizer=self.optimizer,
                            loaded_vars=self.loaded_vars,
                        )

    def load_model_step3(self, **kwargs):
        checkpoint_dir = os.path.join(self.checkpoint_dir, TRAIN_TEMPLATE.format(2))
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)

        load_model_from_hdf5(
            filepath=latest_checkpoint,
            model=self.model,
            optimizer=self.optimizer,
            loaded_vars=self.loaded_vars,
            from_name="branch1_prediction",
            to_name="branch2_prediction",
        )

    def load_model_step4(self, **kwargs):
        checkpoint_dir = os.path.join(self.checkpoint_dir, TRAIN_TEMPLATE.format(2))
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)

        load_model_from_hdf5(
            filepath=latest_checkpoint,
            model=self.model,
            optimizer=self.optimizer,
            loaded_vars=self.loaded_vars,
            from_name="branch2_prediction/dense_1",
            to_name="branch2_prediction/dense",
            **kwargs,
        )

        training_step = TRAIN_TEMPLATE.format(3)
        checkpoint_dir = os.path.join(self.checkpoint_dir, training_step)
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)

        load_model_from_hdf5(
            filepath=latest_checkpoint, model=self.model, optimizer=self.optimizer, loaded_vars=self.loaded_vars
        )

    def load_model_step5(self, **kwargs):
        training_step = TRAIN_TEMPLATE.format(self.step)
        checkpoint_dir = os.path.join(self.checkpoint_dir, training_step)
        if not tf.io.gfile.exists(checkpoint_dir):
            training_step = TRAIN_TEMPLATE.format(4)
            checkpoint_dir = os.path.join(self.checkpoint_dir, training_step)

        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)

        load_model_from_hdf5(
            filepath=latest_checkpoint,
            model=self.model,
            optimizer=self.optimizer,
            loaded_vars=self.loaded_vars,
            to_name="mobilenetv2_stn_base_1.40_224_",
            from_name="mobilenetv2_global_branch_1.40_224_",
        )

        load_model_from_hdf5(
            filepath=latest_checkpoint,
            model=self.model,
            optimizer=self.optimizer,
            loaded_vars=self.loaded_vars,
            to_name="branch2_prediction/dense_1",
            from_name="global_prediction/dense_2",
        )

    def load_model_variables(self, **kwargs):
        training = kwargs.get("training", True)
        if training:
            self.loaded_vars = {}
            if self.step == 1:
                self.load_model_step1(**kwargs)
            elif self.step == 2:
                self.load_model_step2(**kwargs)
            elif self.step == 3:
                self.load_model_step3(**kwargs)
            elif self.step == 4:
                self.load_model_step4(**kwargs)
            elif self.step == 5:
                self.load_model_step5(**kwargs)
        else:
            training_step = TRAIN_TEMPLATE.format(self.step)
            checkpoint_dir = os.path.join(self.checkpoint_dir, training_step)
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
            load_model_from_hdf5(
                filepath=latest_checkpoint, model=self.model, optimizer=self.optimizer, loaded_vars=self.loaded_vars
            )

    def restore_model_from_latest_checkpoint_if_exist(self, **kwargs):
        result = False
        load_from_old_format = False

        step = kwargs.get("training_step", 0)

        if self.checkpoint:
            checkpoint, checkpoint_manager = self.checkpoint
            if checkpoint_manager.latest_checkpoint:
                try:
                    status = checkpoint.restore(checkpoint_manager.latest_checkpoint)
                    status.assert_existing_objects_matched()
                    result = True
                except Exception:
                    logging.info("Can't load checkpoint: %s", checkpoint_manager.latest_checkpoint)
                    logging.info("Continue load from load format file...")
            else:
                load_from_old_format = True

        if step == 1 and load_from_old_format:
            latest_checkpoint = kwargs.pop("checkpoint_dir")
            if latest_checkpoint:
                var_loaded = None
                if self.training:
                    try:
                        var_loaded = Orchids52Mobilenet140.load_from_v1(
                            self,
                            latest_checkpoint=latest_checkpoint,
                            target_model="mobilenetv2_stn_base_1.40_224_",
                            model_name="MobilenetV2",
                            include_prediction_layer=False,
                            **kwargs,
                        )
                    except Exception:
                        logging.info("Can't load checkpoint: %s", latest_checkpoint)
                        logging.info("Continue load from load format file...")
                else:
                    var_loaded = self.load_from_v1(latest_checkpoint, **kwargs)

                if var_loaded:
                    result = load_weight(
                        var_loaded=var_loaded, all_vars=self.model.weights, optimizer=self.model.optimizer, **kwargs
                    )

        return result


def create_orchid_mobilenet_v2_15(num_classes, optimizer=None, loss_fn=None, **kwargs):
    boundary_loss = None
    estimate_block = None
    global_model = None
    shared_model = None
    stn_dense_1 = None
    stn_dense_2 = None
    dropout = kwargs.get("dropout", 0.5)
    training = kwargs.get("training", False)
    step = kwargs.pop("step") if "step" in kwargs else ""
    batch_size = kwargs.pop("batch_size") if "batch_size" in kwargs else 1
    activation = kwargs.pop("activation") if "activation" in kwargs else None

    inputs = keras.Input(shape=IMG_SHAPE_224)
    preprocess_layer = PreprocessLayer()
    stn_base_model = create_mobilenet_v2(
        input_shape=IMG_SHAPE_224, alpha=1.4, include_top=False, weights="imagenet", sub_name="stn_base"
    )

    processed_inputs = preprocess_layer(inputs)

    if step > 1:
        logits_1 = None
        logits_2 = None
        prediction_layer_g = None
        prediction_layer_1 = None
        prediction_layer_2 = None
        bound_errs = []
        stn_out_params = 2

        shared_model = create_mobilenet_v2(
            input_shape=IMG_SHAPE_224, alpha=1.4, include_top=False, weights="imagenet", sub_name="shared_branch"
        )

        stn_logits = stn_base_model(processed_inputs)
        stn_layer = SpatialTransformerNetwork(
            batch_size=batch_size, width=default_image_size, height=default_image_size
        )

        if step == 3 or step >= 4:
            stn_dense_1 = keras.Sequential(
                [
                    Conv2DWrapper(filters=128, kernel_size=[1, 1], activation="relu", name="stn_conv2d_1"),
                    keras.layers.Flatten(),
                    DenseWrapper(units=128, activation="tanh", name="stn_dense_128_1"),
                    keras.layers.Dropout(dropout),
                    FullyConnectedLayer(
                        stn_out_params,
                        kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.4),
                        normalizer_fn=tf.math.l2_normalize,
                        activation="tanh",
                        name="stn_dense_{}_1".format(stn_out_params),
                    ),
                ],
                name="stn_dense1",
            )
            stn_dense_1_output = stn_dense_1(stn_logits)
            stn_output_1, bound_err_1 = stn_layer(
                inputs=processed_inputs, scale=0.8, theta=stn_dense_1_output, training=training
            )

            logit_1 = shared_model(stn_output_1)

            prediction_layer_1 = PredictionLayer(
                num_classes=num_classes, dropout=dropout, activation=None, name="branch1_prediction"
            )
            output_1 = prediction_layer_1(logit_1)

            bound_errs.append(bound_err_1)

        if step == 2 or step >= 4:
            stn_dense_2 = keras.Sequential(
                [
                    Conv2DWrapper(filters=128, kernel_size=[1, 1], activation="relu", name="stn_conv2d_2"),
                    keras.layers.Flatten(),
                    DenseWrapper(units=128, activation="tanh", name="stn_dense_128_2"),
                    keras.layers.Dropout(dropout),
                    FullyConnectedLayer(
                        stn_out_params,
                        kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.4),
                        normalizer_fn=tf.math.l2_normalize,
                        activation="tanh",
                        name="stn_dense_{}_2".format(stn_out_params),
                    ),
                ],
                name="stn_dense2",
            )
            stn_dense_2_output = stn_dense_2(stn_logits)
            stn_output_2, bound_err_2 = stn_layer(
                inputs=processed_inputs, scale=0.5, theta=stn_dense_2_output, training=training
            )

            logit_2 = shared_model(stn_output_2)

            prediction_layer_2 = PredictionLayer(
                num_classes=num_classes, dropout=dropout, activation=None, name="branch2_prediction"
            )
            output_2 = prediction_layer_2(logit_2)

            bound_errs.append(bound_err_2)

        if step == 2:
            outputs = output_2
        if step == 3:
            outputs = output_1

        if step == 4:
            outputs = tf.reduce_mean([output_1, output_2], axis=0)

        if step == 5:
            prediction_layer_g = PredictionLayer(
                num_classes=num_classes, dropout=dropout, activation=None, name="global_prediction"
            )
            global_model = create_mobilenet_v2(
                input_shape=IMG_SHAPE_224, alpha=1.4, include_top=False, weights="imagenet", sub_name="global_branch"
            )
            output_g = global_model(processed_inputs)
            logits_g = prediction_layer_g(output_g)

            estimate_block = EstimationBlock(num_classes=num_classes, batch_size=batch_size)
            outputs = estimate_block([logits_g, output_1, output_2])

        if len(bound_errs) > 1:
            bound_err = tf.concat(bound_errs, axis=1)
        else:
            bound_err = bound_errs[0]

        if training:
            bound_std = tf.constant(np.full(bound_err.shape, 0.00, dtype=np.float32), name="bound_std_zero")
            boundary_loss = keras.Model(inputs, keras.losses.MSE(bound_err, bound_std), name="mse")

        if activation == "softmax":
            outputs = tf.keras.activations.softmax(outputs)

    else:
        prediction_layer_g = PredictionLayer(num_classes=num_classes, dropout=dropout, activation=None)
        mobilenet_logits = stn_base_model(processed_inputs)
        outputs = prediction_layer_g(mobilenet_logits)

    model = Orchids52Mobilenet140STN(
        inputs,
        outputs,
        optimizer=optimizer,
        loss_fn=loss_fn,
        base_model=stn_base_model,
        stn_denses=[stn_dense_1, stn_dense_2],
        estimate_block=estimate_block,
        predict_models=[prediction_layer_g, prediction_layer_1, prediction_layer_2],
        branch_model=[global_model, shared_model],
        boundary_loss=boundary_loss,
        training=training,
        step=step,
    )
    return model
