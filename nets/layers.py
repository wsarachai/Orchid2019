from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import nets
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras import backend as K
from utils.summary import k_summary
from nets.const_vars import REGULARIZER_L2, default_image_size
from nets.mobilenet_v2 import IMG_SHAPE_224
from tensorflow.python.keras import initializers
from tensorflow.python.keras import activations


@tf.function
def augmentation_m1(inputs, maxval=12):
    sel = tf.random.uniform([], maxval=maxval, dtype=tf.int32)
    return tf.switch_case(
        sel,
        branch_fns={
            0: lambda: tf.image.random_flip_left_right(inputs),
            1: lambda: tf.image.random_flip_up_down(inputs),
            2: lambda: tf.image.rot90(inputs),
        },
        default=lambda: inputs,
    )


@tf.function
def augmentation_m2(inputs, maxval=24):
    sel = tf.random.uniform([], maxval=maxval, dtype=tf.int32)
    return tf.switch_case(
        sel,
        branch_fns={
            0: lambda: tf.image.random_brightness(inputs, max_delta=0.2),
            1: lambda: tf.image.random_contrast(inputs, lower=0.2, upper=0.5),
            2: lambda: tf.image.random_saturation(inputs, lower=1, upper=5),
            3: lambda: tf.image.random_hue(inputs, max_delta=0.2),
        },
        default=lambda: inputs,
    )


class PreprocessLayer(keras.layers.Layer):
    def __init__(self, fast=True):
        super(PreprocessLayer, self).__init__()
        self.fast = fast

    def call(self, inputs, training=None, **kwargs):
        if training is None:
            training = True if K.learning_phase() == 1 else False

        if training:
            inputs = tf.image.adjust_brightness(image=inputs, delta=0.2)
            tf.function(k_summary.image_update).get_concrete_function(name="image/input", unit=inputs, max_outputs=1)(
                unit=inputs
            )

        # if inputs.dtype != tf.float32:
        #     inputs = tf.image.convert_image_dtype(inputs, dtype=tf.float32)

        # image_resize = tf.multiply(inputs, 1.0 / 255.0)
        # image_resize = tf.subtract(image_resize, 0.5)
        # inputs = tf.multiply(image_resize, 2.0)

        # if training:
        #     if tf.math.equal(self.fast, 1):
        #         inputs = augmentation_m1(inputs)
        #     else:
        #         inputs = augmentation_m1(inputs)
        #         inputs = augmentation_m2(inputs)

        #     tf.function(k_summary.image_update).get_concrete_function(
        #         name="image/input/preprocess", unit=inputs, max_outputs=1
        #     )(unit=inputs)

        return inputs


class ADropout(keras.layers.Layer):
    def __init__(self, overfitting, **kwargs):
        super(ADropout, self).__init__(**kwargs)
        self.overfitting = overfitting
        self.dropout2 = keras.layers.Dropout(0.2)
        self.dropout8 = keras.layers.Dropout(0.8)

    def call(self, inputs, training=None, **kwargs):
        if training is None:
            training = True if K.learning_phase() == 1 else False

        if tf.math.less_equal(self.overfitting, 0.15):
            outputs = self.dropout2(inputs, training=training)
        else:
            outputs = self.dropout8(inputs, training=training)
        return outputs


class PredictionLayer(keras.layers.Layer):
    def __init__(self, num_classes, overfitting, activation=None):
        super(PredictionLayer, self).__init__()
        self.global_average_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout = ADropout(overfitting=overfitting)

        self.dense = keras.layers.Dense(
            num_classes,
            kernel_initializer=None,
            kernel_regularizer=tf.keras.regularizers.l2(REGULARIZER_L2),
            # activation='elu',
        )
        self.prediction_fn = activations.get(activation)

    def call(self, inputs, training=None, **kwargs):
        if training is None:
            training = True if K.learning_phase() == 1 else False

        inputs = self.global_average_pooling(inputs, training=training)
        inputs = self.dropout(inputs, training=training)
        inputs = self.dense(inputs, training=training)
        inputs = self.prediction_fn(inputs)
        if training and self.trainable:
            histogram_update_k = tf.function(k_summary.histogram_update).get_concrete_function(
                "kernel", self.dense.kernel
            )
            histogram_update_b = tf.function(k_summary.histogram_update).get_concrete_function("bias", self.dense.bias)
            histogram_update_k(self.dense.kernel)
            histogram_update_b(self.dense.bias)
        return inputs


class BranchBlock(keras.layers.Layer):
    def __init__(self, num_classes, batch_size, overfitting, width=default_image_size, height=default_image_size):
        super(BranchBlock, self).__init__()
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.global_branch_model = nets.mobilenet_v2.create_mobilenet_v2(
            input_shape=IMG_SHAPE_224, alpha=1.4, include_top=False, weights="imagenet", sub_name="global_branch"
        )
        self.shared_branch_model = nets.mobilenet_v2.create_mobilenet_v2(
            input_shape=IMG_SHAPE_224, alpha=1.4, include_top=False, weights="imagenet", sub_name="shared_branch"
        )
        self.branches_prediction_models = [
            PredictionLayer(num_classes=num_classes, overfitting=overfitting),
            PredictionLayer(num_classes=num_classes, overfitting=overfitting),
            PredictionLayer(num_classes=num_classes, overfitting=overfitting),
        ]

    def call(self, inputs, training=None, **kwargs):
        if training is None:
            training = True if K.learning_phase() == 1 else False

        return [
            self.get_prediction_layer(1, inputs[0], self.branches_prediction_models[0], training),
            self.get_prediction_layer(2, inputs[1], self.branches_prediction_models[1], training),
            self.get_prediction_layer(2, inputs[2], self.branches_prediction_models[2], training),
        ]

    def get_prediction_layer(self, branch, inp, prediction, training):
        if branch == 1:
            x = self.global_branch_model(inp, training=training)
            x = prediction(x, training=training)
        else:
            x = self.shared_branch_model(inp, training=training)
            x = prediction(x, training=training)
        return x

    def set_trainable_for_global_branch(self, trainable, fine_tune_at=100):
        if trainable:
            self.global_branch_model.trainable = True

            if fine_tune_at:
                # Freeze all the layers before the `fine_tune_at` layer
                for layer in self.global_branch_model.layers[:fine_tune_at]:
                    layer.trainable = False
        else:
            self.global_branch_model.trainable = False

    def set_trainable_for_share_branch(self, trainable, fine_tune_at=100):
        if trainable:
            self.shared_branch_model.trainable = True

            if fine_tune_at:
                # Freeze all the layers before the `fine_tune_at` layer
                for layer in self.shared_branch_model.layers[:fine_tune_at]:
                    layer.trainable = False
        else:
            self.shared_branch_model.trainable = False


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

    def call(self, inputs, training=None, **kwargs):
        if training is None:
            training = True if K.learning_phase() == 1 else False

        main_net = c_t = inputs[0]

        input_and_hstate_concatenated = tf.concat(values=[c_t, inputs[1]], axis=1)
        c_t = self.dense(input_and_hstate_concatenated)
        c_t = self.batch_norm(c_t)
        main_net = main_net + c_t

        input_and_hstate_concatenated = tf.concat(values=[c_t, inputs[2]], axis=1)
        c_t = self.dense(input_and_hstate_concatenated)
        c_t = self.batch_norm(c_t)
        main_net = main_net + c_t

        if training and self.trainable:
            histogram_update = tf.function(k_summary.histogram_update).get_concrete_function(
                "kernel", self.dense.kernel
            )
            histogram_update(self.dense.kernel)

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

    def call(self, inputs, training=None, **kwargs):
        if training is None:
            training = True if K.learning_phase() == 1 else False

        if training and self.trainable:
            k_summary.histogram_update("kernel", self.kernel)

        x = tf.matmul(inputs, self.kernel)
        if self.normalizer_fn is not None:
            x = self.normalizer_fn(x)
        return self.activation(x)


class PrintingNode(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PrintingNode, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return tf.compat.v1.Print(inputs, [inputs])


class Conv2DWrapper(keras.layers.Conv2D):
    def __init__(self, **kwargs):
        super(Conv2DWrapper, self).__init__(**kwargs)

    def call(self, inputs):
        if self.trainable:
            k_summary.histogram_update("kernel", self.kernel)
            k_summary.histogram_update("bias", self.bias)

        return super(Conv2DWrapper, self).call(inputs)


class DenseWrapper(keras.layers.Dense):
    def __init__(self, **kwargs):
        super(DenseWrapper, self).__init__(**kwargs)

    def call(self, inputs):
        if self.trainable:
            k_summary.histogram_update("kernel", self.kernel)
            k_summary.histogram_update("bias", self.bias)

        return super(DenseWrapper, self).call(inputs)
