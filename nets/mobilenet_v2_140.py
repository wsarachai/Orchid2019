from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import nets
import lib_utils
import tensorflow as tf
import tensorflow.keras as keras

from nets.mobilenet_v2 import create_mobilenet_v2, IMG_SHAPE_224


class Orchids52Mobilenet140(keras.Model):
    def __init__(self, inputs, outputs,
                 base_model,
                 predict_models,
                 training,
                 step):
        super(Orchids52Mobilenet140, self).__init__(inputs, outputs, trainable=training)
        self.base_model = base_model
        self.predict_models = predict_models
        self.training = training
        self.step = step

    def call(self, inputs, training=None, mask=None):
        return super(Orchids52Mobilenet140, self).call(inputs, training, mask)

    def get_config(self):
        return super(Orchids52Mobilenet140, self).get_config()

    def set_mobilenet_training_status(self, trainable):
        if self.base_model:
            self.base_model.trainable = trainable

    def set_prediction_training_status(self, trainable):
        if self.predict_models:
            self.predict_models.trainable = trainable
            for p in self.predict_models:
                p.trainable = trainable

    def config_layers(self, step):
        import nets
        if step == nets.utils.TRAIN_STEP1:
            self.set_mobilenet_training_status(False)

    def save_model_weights(self, filepath, epoch, overwrite=True, save_format=None):
        model_path = os.path.join(filepath, self.step)
        if not tf.io.gfile.exists(model_path):
            os.makedirs(model_path)
        super(Orchids52Mobilenet140, self).save_weights(filepath=lib_utils.get_checkpoint_file(model_path, epoch),
                                                        overwrite=overwrite,
                                                        save_format=save_format)
        if self.base_model:
            base_model_path = os.path.join(filepath, 'base_model')
            if not tf.io.gfile.exists(base_model_path):
                os.makedirs(base_model_path)
            self.base_model.save_weights(filepath=lib_utils.get_checkpoint_file(base_model_path, epoch),
                                         overwrite=overwrite,
                                         save_format=save_format)

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
        super(Orchids52Mobilenet140, self).save(filepath=model_path,
                                                overwrite=overwrite,
                                                include_optimizer=include_optimizer,
                                                save_format=save_format,
                                                signatures=signatures,
                                                options=options)

    def load_model_step1(self, filepath, epoch, by_name=False, skip_mismatch=False):
        predict_model_path = os.path.join(filepath, 'predict_model', '00')
        if tf.io.gfile.exists(predict_model_path):
            for m in self.predict_models:
                m.load_weights(filepath=lib_utils.get_checkpoint_file(predict_model_path, epoch),
                               by_name=by_name,
                               skip_mismatch=skip_mismatch)

    def load_model_weights(self,
                           checkpoint_path,
                           epoch,
                           by_name=False,
                           skip_mismatch=False):
        self.config_layers(self.step)
        if self.step == nets.utils.TRAIN_STEP1:
            self.load_model_step1(checkpoint_path, epoch, by_name, skip_mismatch)

    def resume_model_weights(self, filepath, by_name=False, skip_mismatch=False):
        self.config_layers(self.step)
        if not hasattr(filepath, 'endswith'):
            filepath = str(filepath)
        super(Orchids52Mobilenet140, self).load_weights(
            filepath=filepath, by_name=by_name, skip_mismatch=skip_mismatch)


def create_mobilenet_v2_14(num_classes,
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
                                  base_model=base_model,
                                  predict_models=[prediction_layer],
                                  training=training,
                                  step=step)

    return model
