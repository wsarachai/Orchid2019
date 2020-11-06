from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Sequential

import nets
import tensorflow.keras as keras
from tensorflow.keras import layers

from lib_utils import get_checkpoint_file, latest_checkpoint
from nets.mobilenet_v2 import default_image_size, create_mobilenet_v2, IMG_SHAPE_224
from stn import pre_spatial_transformer_network

logging = tf.compat.v1.logging


class Orchids52Mobilenet140(keras.Model):
    def __init__(self, inputs, outputs,
                 base_model,
                 stn_dense,
                 predict_models,
                 branch_model,
                 is_training,
                 step):
        super(Orchids52Mobilenet140, self).__init__(inputs, outputs, trainable=is_training)
        self.base_model = base_model
        self.stn_dense = stn_dense
        self.predict_models = predict_models
        self.branch_model = branch_model
        self.is_training = is_training
        self.step = step

    def set_mobilenet_training_status(self, trainable):
        if self.base_model:
            self.base_model.trainable = trainable
        if self.branch_model:
            self.branch_model.trainable = trainable

    def config_layers(self, step):
        import nets
        if step == nets.nets_utils.TRAIN_STEP1:
            self.set_mobilenet_training_status(False)
        elif step == nets.nets_utils.TRAIN_STEP2:
            self.set_mobilenet_training_status(False)
            self.predict_models.trainable = False
        elif step == nets.nets_utils.TRAIN_STEP3:
            self.set_mobilenet_training_status(False)
            self.stn_dense.trainable = False
            self.predict_models.trainable = False
        elif step == nets.nets_utils.TRAIN_V2_STEP1:
            self.set_mobilenet_training_status(False)
        elif step == nets.nets_utils.TRAIN_V2_STEP2:
            self.set_mobilenet_training_status(True)

    def save_model_weights(self, filepath, epoch, overwrite=True, save_format=None):
        model_path = os.path.join(filepath, self.step)
        if not tf.io.gfile.exists(model_path):
            tf.io.gfile.mkdir(model_path)
        super(Orchids52Mobilenet140, self).save_weights(filepath=get_checkpoint_file(model_path, epoch),
                                                        overwrite=overwrite,
                                                        save_format=save_format)
        if self.base_model:
            base_model_path = os.path.join(filepath, 'base_model')
            if not tf.io.gfile.exists(base_model_path):
                tf.io.gfile.mkdir(base_model_path)
            self.base_model.save_weights(filepath=get_checkpoint_file(base_model_path, epoch),
                                         overwrite=overwrite,
                                         save_format=save_format)
        if self.branch_model:
            branch_model_path = os.path.join(filepath, 'branch_model')
            if not tf.io.gfile.exists(branch_model_path):
                tf.io.gfile.mkdir(branch_model_path)
            self.branch_model.save_weights(filepath=get_checkpoint_file(branch_model_path, epoch),
                                           overwrite=overwrite,
                                           save_format=save_format)
        if self.predict_models:
            predict_model_path = os.path.join(filepath, 'predict_model')
            if not tf.io.gfile.exists(predict_model_path):
                tf.io.gfile.mkdir(predict_model_path)
            for k, m in enumerate(self.predict_models):
                model_path = os.path.join(predict_model_path, '{:02d}'.format(k))
                if not tf.io.gfile.exists(model_path):
                    tf.io.gfile.mkdir(model_path)
                m.save_weights(filepath=get_checkpoint_file(model_path, epoch),
                               overwrite=overwrite,
                               save_format=save_format)

    def load_model_step2(self, filepath, epoch, by_name=False, skip_mismatch=False):
        base_model_path = os.path.join(filepath, 'base_model')
        self.base_model.load_weights(filepath=get_checkpoint_file(base_model_path, epoch),
                                     by_name=by_name,
                                     skip_mismatch=skip_mismatch)
        predict_model_path = os.path.join(filepath, 'predict_model', '00')
        for m in self.predict_models:
            m.load_weights(filepath=get_checkpoint_file(predict_model_path, epoch),
                           by_name=by_name,
                           skip_mismatch=skip_mismatch)

    def load_model_step3(self, filepath, epoch, by_name=False, skip_mismatch=False):
        base_model_path = os.path.join(filepath, 'base_model')
        self.base_model.load_weights(filepath=get_checkpoint_file(base_model_path, epoch),
                                     by_name=by_name,
                                     skip_mismatch=skip_mismatch)
        base_model_path = os.path.join(filepath, 'branch_model')
        self.branch_model.load_weights(filepath=get_checkpoint_file(base_model_path, epoch),
                                       by_name=by_name,
                                       skip_mismatch=skip_mismatch)
        for k, m in enumerate(self.predict_models):
            predict_model_path = os.path.join(filepath, 'predict_model', '{:02d}'.format(k))
            m.load_weights(filepath=get_checkpoint_file(predict_model_path, epoch),
                           by_name=by_name,
                           skip_mismatch=skip_mismatch)

    def load_model_weights(self,
                           checkpoint_path,
                           epoch,
                           by_name=False,
                           skip_mismatch=False):
        self.config_layers(self.step)
        if self.step == nets.nets_utils.TRAIN_STEP2:
            self.load_model_step2(checkpoint_path, epoch, by_name, skip_mismatch)
        elif self.step == nets.nets_utils.TRAIN_STEP3:
            self.load_model_step3(checkpoint_path, epoch, by_name, skip_mismatch)
        elif self.step == nets.nets_utils.TRAIN_STEP4:
            latest, _ = latest_checkpoint(checkpoint_path, nets.nets_utils.TRAIN_STEP3)
            super(Orchids52Mobilenet140, self).load_weights(
                filepath=latest,
                by_name=by_name,
                skip_mismatch=skip_mismatch)
        elif self.step == nets.nets_utils.TRAIN_V2_STEP2:
            self.load_model_v2_step2(checkpoint_path, epoch, by_name, skip_mismatch)
        else:
            filepath = os.path.join(checkpoint_path, self.step)
            filepath = get_checkpoint_file(filepath, epoch=epoch)
            if tf.io.gfile.exists(filepath):
                super(Orchids52Mobilenet140, self).load_weights(
                    filepath=filepath,
                    by_name=by_name,
                    skip_mismatch=skip_mismatch)

    def resume_model_weights(self, filepath, by_name=False, skip_mismatch=False):
        self.config_layers(self.step)
        if not hasattr(filepath, 'endswith'):
            filepath = str(filepath)
        super(Orchids52Mobilenet140, self).load_weights(
            filepath=filepath, by_name=by_name, skip_mismatch=skip_mismatch)


def create_predict_module(num_classes, name):
    module = Sequential([
        keras.layers.GlobalAveragePooling2D(name='{}_global'.format(name)),
        layers.Dropout(0.2, name='{}_droupout'.format(name)),
        layers.Dense(128, activation='relu', name='{}_fc_128'.format(name)),
        layers.Dense(num_classes, activation='relu', name='{}_fc'.format(name))
    ])
    return module


def create_orchid_mobilenet_v2_14(num_classes,
                                  is_training=False,
                                  **kwargs):
    import nets

    if 'step' in kwargs:
        step = kwargs.pop('step')
    else:
        step = ''

    stn_dense = None
    branch_base_model = None
    branches_prediction_models = []
    data_augmentation = keras.Sequential([
        keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])
    preprocess_input = keras.applications.mobilenet_v2.preprocess_input
    batch_size = kwargs.pop('batch_size')

    inputs = keras.Input(shape=(default_image_size, default_image_size, 3), batch_size=batch_size)
    inputs = data_augmentation(inputs)
    inputs = preprocess_input(inputs)

    # Create the base model from the pre-trained model MobileNet V2
    stn_base_model = create_mobilenet_v2(input_tensor=inputs,
                                         alpha=1.4,
                                         include_top=False,
                                         weights='imagenet',
                                         sub_name='01')

    if step != nets.nets_utils.TRAIN_STEP1:
        scales = [0.8, 0.6]
        element_size = 3  # [x, y, scale]
        fc_num = element_size * len(scales)
        stn_dense = Sequential([
            layers.Conv2D(128, [1, 1], activation='relu', name="t2_stn_conv2d_resize_128"),
            layers.Flatten(name='t2_stn_flatten'),
            keras.layers.Dropout(rate=0.2, name='t2_stn_dropout'),
            layers.Dense(128, activation='relu', name='t2_stn_dense_128'),
            layers.Dense(fc_num, activation='tanh', activity_regularizer='l2', name='t1_dense_6')
        ])

        x = stn_base_model(inputs)
        h_fc1 = stn_dense(x)

        stn_inputs, bound_err = pre_spatial_transformer_network(inputs,
                                                                h_fc1,
                                                                width=default_image_size,
                                                                height=default_image_size,
                                                                scales=scales)

        if is_training:
            _len = bound_err.get_shape().as_list()[0]
            bound_std = tf.constant(np.full(_len, 0.00, dtype=np.float32))
            tf.losses.mean_squared_error(bound_err, bound_std)

        all_images = []
        for img in stn_inputs:
            all_images.append(img)

        branch_base_model = create_mobilenet_v2(input_shape=IMG_SHAPE_224,
                                                alpha=1.4,
                                                include_top=False,
                                                weights='imagenet',
                                                sub_name='02')
        all_logits = []
        x = branch_base_model(inputs, training=False)
        all_logits.append(x)
        for i, input_image in enumerate(all_images):
            x = branch_base_model(input_image, training=False)
            all_logits.append(x)

        all_predicts = []
        for i, net in enumerate(all_logits):
            branches_prediction = create_predict_module(num_classes=num_classes, name='t2_{i:02d}'.format(i=i))
            x = branches_prediction(net)
            branches_prediction_models.append(branches_prediction)
            all_predicts.append(x)

        if step == nets.nets_utils.TRAIN_STEP2:
            outputs = tf.add_n(all_predicts)
            outputs = tf.divide(outputs, len(all_predicts))
        else:
            # estimation block
            c_t = None
            main_net = None
            for i, net in enumerate(all_predicts):
                if i > 0:
                    input_and_hstate_concatenated = tf.concat(
                        axis=1,
                        values=[c_t, net],
                        name='t2_concat_{i:02d}'.format(i=i))
                    c_t = layers.Dense(num_classes,
                                       name='t2_Dense_{i:02d}'.format(i=i)
                                       )(input_and_hstate_concatenated)
                    main_net = main_net + c_t
                else:
                    main_net = net
                    c_t = net
            outputs = main_net

    else:
        predict_model = create_predict_module(num_classes=num_classes, name='t1')
        branches_prediction_models.append(predict_model)
        x = stn_base_model(inputs)
        outputs = predict_model(x)

    model = Orchids52Mobilenet140(inputs, outputs,
                                  base_model=stn_base_model,
                                  stn_dense=stn_dense,
                                  predict_models=branches_prediction_models,
                                  branch_model=branch_base_model,
                                  is_training=is_training,
                                  step=step)
    return model
