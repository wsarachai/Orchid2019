from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import nets
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras import Sequential
from nets.mobilenet_v2 import default_image_size, create_mobilenet_v2, IMG_SHAPE_224
from stn import pre_spatial_transformer_network

logging = tf.compat.v1.logging


class Orchids52Mobilenet140STN(nets.mobilenet_v2_140.Orchids52Mobilenet140):
    def __init__(self, inputs, outputs,
                 optimizer,
                 loss_fn,
                 base_model,
                 stn_dense,
                 predict_models,
                 branch_model,
                 boundary_loss,
                 training,
                 step):
        super(Orchids52Mobilenet140STN, self).__init__(inputs, outputs,
                                                       optimizer,
                                                       loss_fn,
                                                       base_model,
                                                       predict_models,
                                                       training,
                                                       step)
        self.stn_dense = stn_dense
        self.branch_model = branch_model
        self.boundary_loss = boundary_loss

    def set_mobilenet_training_status(self, trainable):
        super(Orchids52Mobilenet140STN, self).set_mobilenet_training_status(trainable)
        if self.branch_model:
            self.branch_model.trainable = trainable

    def config_layers(self):
        import nets
        if self.step == nets.utils.TRAIN_STEP1:
            self.set_mobilenet_training_status(False)
        elif self.step == nets.utils.TRAIN_STEP2:
            self.set_mobilenet_training_status(False)
            self.set_prediction_training_status(False)
        elif self.step == nets.utils.TRAIN_STEP3:
            self.set_mobilenet_training_status(False)
            self.set_prediction_training_status(False)
            self.stn_dense.trainable = False
        elif self.step == nets.utils.TRAIN_V2_STEP1:
            self.set_mobilenet_training_status(False)
        elif self.step == nets.utils.TRAIN_V2_STEP2:
            self.set_mobilenet_training_status(True)

    def load_model_step2(self):
        for idx, p_ck_mgr in enumerate(self.predict_models_checkpoint_managers):
            if p_ck_mgr.latest_checkpoint:
                status = self.predict_models_checkpoints[idx].restore(p_ck_mgr.latest_checkpoint)
                status.assert_existing_objects_matched()

    def load_model_step3(self):
        for idx, p_ck_mgr in enumerate(self.predict_models_checkpoint_managers):
            if p_ck_mgr.latest_checkpoint:
                status = self.predict_models_checkpoints[idx].restore(p_ck_mgr.latest_checkpoint)
                status.assert_existing_objects_matched()

    def load_model_step4(self):
        latest_checkpoint = self.checkpoint_manager.latest_checkpoint
        if latest_checkpoint:
            status = self.checkpoint.restore(latest_checkpoint)
            status.assert_existing_objects_matched()

    def load_model_v2_step2(self):
        pass

    def load_model_variables(self):
        if self.step == nets.utils.TRAIN_STEP1:
            self.load_model_step1()
        elif self.step == nets.utils.TRAIN_STEP2:
            self.load_model_step2()
        elif self.step == nets.utils.TRAIN_STEP3:
            self.load_model_step3()
        elif self.step == nets.utils.TRAIN_STEP4:
            self.load_model_step4()
        elif self.step == nets.utils.TRAIN_V2_STEP2:
            self.load_model_v2_step2()


def create_orchid_mobilenet_v2_14(num_classes,
                                  optimizer,
                                  loss_fn,
                                  training=False,
                                  **kwargs):
    import nets

    stn_dense = None
    branch_base_model = None
    boundary_loss = None
    branches_prediction_models = []
    step = kwargs.pop('step') if 'step' in kwargs else ''
    batch_size = kwargs.pop('batch_size') if 'batch_size' in kwargs else 32
    model_name = 'orchids52-en'

    data_augmentation = keras.Sequential([
        keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ], name='{}_data-augmentation'.format(model_name))
    preprocess_input = keras.applications.mobilenet_v2.preprocess_input

    inputs = keras.Input(shape=IMG_SHAPE_224, name='{}_input'.format(model_name), batch_size=batch_size)
    aug_inputs = data_augmentation(inputs, training=training)
    preprocess_inputs = preprocess_input(aug_inputs)

    stn_base_model = create_mobilenet_v2(input_shape=IMG_SHAPE_224,
                                         alpha=1.4,
                                         include_top=False,
                                         weights='imagenet',
                                         sub_name='01')

    if step != nets.utils.TRAIN_STEP1:
        scales = [0.8, 0.6]
        element_size = 3  # [x, y, scale]
        fc_num = element_size * len(scales)
        stn_dense = Sequential([
            keras.layers.Conv2D(128, [1, 1], activation='relu', name="t2_stn_conv2d_resize_128"),
            keras.layers.Flatten(name='t2_stn_flatten'),
            keras.layers.Dense(128, name='t2_stn_dense_128'),
            keras.layers.Dropout(rate=0.2, name='t2_stn_dropout'),
            keras.layers.Dense(fc_num, activation='tanh', activity_regularizer='l2', name='t1_dense_6')
        ], name='{}_stn_dense'.format(model_name))

        stn_module = Sequential([
            stn_base_model,
            stn_dense
        ], name='{}_stn_module'.format(model_name))

        stn_fc = stn_module(preprocess_inputs)

        stn_output, bound_err = pre_spatial_transformer_network(preprocess_inputs,
                                                                stn_fc,
                                                                batch_size=batch_size,
                                                                width=default_image_size,
                                                                height=default_image_size,
                                                                scales=scales)

        if training:
            _len = bound_err.shape[0]
            bound_std = tf.constant(np.full(_len, 0.00, dtype=np.float32),
                                    name='{}_bound_std'.format(model_name))
            boundary_loss = keras.Model(inputs,
                                        tf.keras.losses.MSE(bound_err, bound_std),
                                        name='{}_mse'.format(model_name))

        all_images = []
        for img in stn_output:
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
            branches_prediction = nets.utils.create_predict_module(num_classes=num_classes,
                                                                   name='{n}_t2_{i:02d}'.format(n=model_name, i=i))
            x = branches_prediction(net)
            branches_prediction_models.append(branches_prediction)
            all_predicts.append(x)

        if step == nets.utils.TRAIN_STEP2:
            outputs = tf.add_n(all_predicts, name='{}_add_n'.format(model_name))
            num_of_predicts = tf.convert_to_tensor(len(all_predicts), dtype=tf.float32, name='num_of_predicts')
            outputs = tf.divide(outputs, num_of_predicts, name='{}_div'.format(model_name))
        else:
            # estimation block
            c_t = None
            main_net = None
            for i, net in enumerate(all_predicts):
                if i > 0:
                    input_and_hstate_concatenated = tf.concat(
                        axis=1,
                        values=[c_t, net],
                        name='{n}_t2_concat_{i:02d}'.format(n=model_name, i=i))
                    c_t = keras.layers.Dense(num_classes,
                                             name='{n}_t2_Dense_{i:02d}'.format(n=model_name, i=i)
                                             )(input_and_hstate_concatenated)
                    main_net = main_net + c_t
                else:
                    main_net = net
                    c_t = net
            outputs = main_net

    else:
        prediction_layer = nets.utils.create_predict_module(num_classes=num_classes,
                                                            name='mobilenet_v2_14_orchids52',
                                                            activation='softmax')
        branches_prediction_models.append(prediction_layer)

        stn_output = stn_base_model(preprocess_inputs, training=training)
        outputs = prediction_layer(stn_output, training=training)

    model = Orchids52Mobilenet140STN(inputs, outputs,
                                     optimizer=optimizer,
                                     loss_fn=loss_fn,
                                     base_model=stn_base_model,
                                     stn_dense=stn_dense,
                                     predict_models=branches_prediction_models,
                                     branch_model=branch_base_model,
                                     boundary_loss=boundary_loss,
                                     training=training,
                                     step=step)
    return model


def _handle_boundary_loss(name, variable, error_fn):
    def _loss_for_boundary(v):
        with tf.name_scope(name + '/boundary_loss'):
            regularization = error_fn(v)
        return regularization

    return functools.partial(_loss_for_boundary, variable)
