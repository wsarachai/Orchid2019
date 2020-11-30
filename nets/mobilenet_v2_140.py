from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import zipfile
import nets
import lib_utils
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.utils import data_utils

from nets.constants import TRAIN_TEMPLATE
from nets.mobilenet_v2 import PredictionLayer, default_image_size

ORCHIDS_BASE_WEIGHT_PATH = 'https://ndownloader.figshare.com/files'
WEIGHT_MOBILENET_V2 = 'orchids52_mobilenet_v2'
WEIGHT_MOBILENET_V2_TOP = 'orchids52_mobilenet_v2_top'
logging = tf.compat.v1.logging


def load_from_v1(latest_checkpoint):
    # Try reading on v1 data format
    reader = tf.compat.v1.train.NewCheckpointReader(latest_checkpoint)
    var_to_shape_map = reader.get_variable_to_shape_map()
    value_to_load = []
    key_to_numpy = {}
    for key in sorted(var_to_shape_map.items()):
        key_to_numpy.update({key[0]: key[1]})

    keys = [
        'weights',
        'depthwise_weights',
        'BatchNorm/beta',
        'BatchNorm/gamma',
        'BatchNorm/moving_mean',
        'BatchNorm/moving_variance'
    ]
    expds = [
        'expand', 'depthwise', 'project'
    ]

    for k1 in keys:
        for _key in key_to_numpy:
            if _key.startswith('MobilenetV2/Conv/{}'.format(k1)):
                key_to_numpy.pop(_key)
                value = reader.get_tensor(_key)
                value_to_load.append(value)
                break

    for i in range(0, 17):
        for sub in expds:
            for k2 in keys:
                if i == 0:
                    s_search = 'MobilenetV2/expanded_conv/{}/{}'.format(sub, k2)
                else:
                    s_search = 'MobilenetV2/expanded_conv_{}/{}/{}'.format(i, sub, k2)
                for _key in key_to_numpy:
                    if _key.startswith(s_search):
                        key_to_numpy.pop(_key)
                        value = reader.get_tensor(_key)
                        value_to_load.append(value)
                        break

    for k1 in keys:
        for _key in key_to_numpy:
            if _key.startswith('MobilenetV2/Conv_1/{}'.format(k1)):
                key_to_numpy.pop(_key)
                value = reader.get_tensor(_key)
                value_to_load.append(value)
                break

    for s_search in ['MobilenetV2/Logits/Conv2d_1c_1x1/weights',
                     'MobilenetV2/Logits/Conv2d_1c_1x1/biases']:
        for _key in key_to_numpy:
            if _key == s_search:
                key_to_numpy.pop(_key)
                value = reader.get_tensor(_key)
                value_to_load.append(value)
                break

    for _key in key_to_numpy:
        value = reader.get_tensor(_key)
        value_to_load.append(value)

    return value_to_load


def load_trained_weights(model,
                         weights,
                         alpha=1.4,
                         include_top=True):
    model_name = None
    weight_path = None

    if weights == WEIGHT_MOBILENET_V2:
        if alpha == 1.4 and include_top:
            weight_path = ORCHIDS_BASE_WEIGHT_PATH + '/25623764'
            model_name = weights + '_' + str(alpha)
        elif alpha == 1.4 and not include_top:
            weight_path = ORCHIDS_BASE_WEIGHT_PATH + '/25623752'
            model_name = weights + '_' + str(alpha) + '_no_top'
    elif weights == WEIGHT_MOBILENET_V2_TOP:
        weight_path = ORCHIDS_BASE_WEIGHT_PATH + '/25623755'
        model_name = weights + '_' + str(alpha) + '_top'

    assert model_name and weight_path

    weights_path = data_utils.get_file(
        model_name, weight_path, cache_subdir='models')

    model_checkpoint_files = os.path.join(weights_path + '_weights')
    if not tf.io.gfile.exists(model_checkpoint_files):
        with zipfile.ZipFile(weights_path, 'r') as zip_ref:
            model_checkpoint_path = os.path.join(weights_path + '_weights')
            zip_ref.extractall(model_checkpoint_path)
    if tf.io.gfile.exists(model_checkpoint_files):
        model_checkpoint_files = os.path.join(model_checkpoint_files, 'chk')
        model.load_weights(model_checkpoint_files)


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
        self.alpha = 1.4
        self.max_to_keep = 5

        self.checkpoint_path = None
        self.checkpoint = None
        self.prediction_layer_checkpoints = []

    def compile(self, **kwargs):
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss_fn,
                           **kwargs)

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
        self.checkpoint_path = checkpoint_path
        checkpoint = tf.train.Checkpoint(
            step=tf.Variable(1),
            optimizer=self.optimizer,
            model=self.model)
        checkpoint_prefix = os.path.join(checkpoint_path, self.step)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, directory=checkpoint_prefix, max_to_keep=self.max_to_keep)
        self.checkpoint = (checkpoint, checkpoint_manager)

        predict_layers_path = os.path.join(checkpoint_path, 'predict_layers')
        for idx, predict_layer in enumerate(self.predict_layers):
            prediction_layer_prefix = lib_utils.get_checkpoint_file(predict_layers_path, idx)
            self.prediction_layer_checkpoints.append((predict_layer, prediction_layer_prefix))

    def save_model_variables(self):
        _, checkpoint_manager = self.checkpoint
        checkpoint_manager.save()
        for checkpoint in self.prediction_layer_checkpoints:
            predict_layer, save_dir = checkpoint
            predict_layer.save_weights(save_dir)

    def load_trained_weights(self,
                             weights=WEIGHT_MOBILENET_V2,
                             alpha=1.4,
                             include_top=True):
        load_trained_weights(self.model,
                             weights=weights,
                             alpha=alpha,
                             include_top=include_top)

    def restore_model_from_latest_checkpoint_if_exist(self):
        result = False
        if self.checkpoint:
            checkpoint, checkpoint_manager = self.checkpoint
            if checkpoint_manager.latest_checkpoint:
                logging.info("Loading weight from [{}]".format(checkpoint_manager.latest_checkpoint))
                status = checkpoint.restore(checkpoint_manager.latest_checkpoint)
                status.assert_existing_objects_matched()
                result = True
        return result

    def get_step_number_from_latest_checkpoint(self):
        try:
            _, checkpoint_manager = self.checkpoint
            index = checkpoint_manager.latest_checkpoint.index('ckpt-')
            step = checkpoint_manager.latest_checkpoint[index:][5:]
            step = int(step)
        except:
            return 1
        else:
            return step

    def load_weights(self, checkpoint_path):
        status = self.model.load_weights(checkpoint_path)
        status.assert_consumed()

    def save_weight(self, checkpoint_path):
        self.model.save_weights(checkpoint_path)

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
        if self.step == nets.constants.TRAIN_STEP1:
            self.set_mobilenet_training_status(False)

    def load_model_variables(self):
        if self.step == nets.constants.TRAIN_STEP1:
            self.load_model_step1()

    def load_model_step1(self):
        load_trained_weights(self.mobilenet,
                             weights=WEIGHT_MOBILENET_V2,
                             alpha=self.alpha,
                             include_top=False)
        for predict_layer in self.predict_layers:
            load_trained_weights(predict_layer,
                                 weights=WEIGHT_MOBILENET_V2_TOP)

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


def create_mobilenet_v2_140(num_classes,
                            optimizer,
                            loss_fn,
                            training=False,
                            include_top=True,
                            **kwargs):
    alpha=1.4
    step = kwargs.pop('step') if 'step' in kwargs else TRAIN_TEMPLATE.format(step=1)

    prediction_layers = []
    inputs = keras.Input(shape=nets.mobilenet_v2.IMG_SHAPE_224, dtype=tf.float32)
    mobilenet = nets.mobilenet_v2.create_mobilenet_v2_custom(
        input_shape=nets.mobilenet_v2.IMG_SHAPE_224,
        alpha=alpha,
        include_top=False,
        classes=num_classes)

    outputs = mobilenet(inputs, training=training)
    if include_top:
        prediction_layer = PredictionLayer(num_classes=num_classes)
        outputs = prediction_layer(outputs, training=training)
        prediction_layers.append(prediction_layer)

    model = Orchids52Mobilenet140(inputs, outputs,
                                  optimizer=optimizer,
                                  loss_fn=loss_fn,
                                  mobilenet=mobilenet,
                                  predict_layers=prediction_layers,
                                  training=training,
                                  step=step)

    if include_top:
        model.load_trained_weights(alpha=alpha, include_top=include_top)
    else:
        model.load_trained_weights(alpha=alpha, include_top=include_top)

    return model


create_mobilenet_v2_140.width = default_image_size
create_mobilenet_v2_140.height = default_image_size
