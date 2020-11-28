from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.keras import layers, Sequential

from preprocesing.inception_preprocessing import preprocess_image

logging = tf.compat.v1.logging

BASE_WEIGHT_PATH = ('https://storage.googleapis.com/tensorflow/'
                    'keras-applications/mobilenet_v2/')
default_image_size = 224
IMG_SIZE_224 = (default_image_size, default_image_size)
IMG_SHAPE_224 = IMG_SIZE_224 + (3,)


class PreprocessLayer(layers.Layer):
    def __init__(self, width, height, fast_mode=True):
        super(PreprocessLayer, self).__init__()
        self.width = width
        self.height = height
        self.fast_mode = fast_mode

    def call(self, inputs, **kwargs):
        is_training = kwargs.pop('training', False)
        is_training = False if is_training is None else is_training
        inputs = preprocess_image(inputs,
                                  width=self.width,
                                  height=self.height,
                                  is_training=is_training,
                                  fast_mode=self.fast_mode)

        return inputs


class PredictionLayer(layers.Layer):
    def __init__(self, num_classes, dropout_ratio=0.2, activation="softmax"):
        super(PredictionLayer, self).__init__()
        self.activation = activation
        self.dropout = layers.Dropout(dropout_ratio)
        self.dense = layers.Conv2D(
            num_classes,
            kernel_size=1,
            padding='same',
            use_bias=True,
            activation=None,
            bias_initializer=tf.zeros_initializer(),
            name='dense-{}'.format(num_classes))

    def call(self, inputs, **kwargs):
        _training = kwargs.pop('training')
        inputs = global_pool(inputs)
        inputs = self.dropout(inputs, training=_training)
        inputs = self.dense(inputs, training=_training)
        inputs = tf.squeeze(inputs, [1, 2])
        if self.activation == "softmax":
            inputs = tf.nn.softmax(inputs, axis=1)
        return inputs


def _inverted_res_block(name, inputs, expansion, stride, alpha, filters, block_id):
    """Inverted ResNet block."""
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    in_channels = backend.int_shape(inputs)[channel_axis]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = '{}_block_{}_'.format(name, block_id)

    if block_id:
        # Expand
        x = layers.Conv2D(
            expansion * in_channels,
            kernel_size=1,
            padding='same',
            use_bias=False,
            activation=None,
            name=prefix + 'expand')(
            x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + 'expand_BN')(
            x)
        x = layers.ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = '{}_expanded_conv_'.format(name)

    # Depthwise
    if stride == 2:
        x = layers.ZeroPadding2D(
            padding=imagenet_utils.correct_pad(x, 3),
            name=prefix + 'pad')(x)
    x = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        activation=None,
        use_bias=False,
        padding='same' if stride == 1 else 'valid',
        name=prefix + 'depthwise')(
        x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'depthwise_BN')(
        x)

    x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = layers.Conv2D(
        pointwise_filters,
        kernel_size=1,
        padding='same',
        use_bias=False,
        activation=None,
        name=prefix + 'project')(
        x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'project_BN')(
        x)

    if in_channels == pointwise_filters and stride == 1:
        return layers.Add(name=prefix + 'add')([inputs, x])
    return x


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def global_pool(input_tensor, pool_op=layers.AvgPool2D):
    shape = input_tensor.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        pool_size = tf.convert_to_tensor(
            [tf.shape(input_tensor)[1], tf.shape(input_tensor)[2]])
    else:
        pool_size = [shape[1], shape[2]]
    output = pool_op(pool_size=pool_size, strides=[1, 1], padding='valid')(input_tensor)
    output.set_shape([None, 1, 1, None])
    return output


def create_mobilenet_v2_custom(input_shape,
                               alpha,
                               classes,
                               include_top=False,
                               model_name='mobilenet_v2',
                               suffix_name=None):
    if suffix_name is not None:
        model_name = "{}-{}".format(model_name, suffix_name)

    inputs = layers.Input(shape=input_shape)
    channel_axis = -1
    first_block_filters = _make_divisible(32 * alpha, 8)
    x = layers.Conv2D(
        first_block_filters,
        kernel_size=3,
        strides=(2, 2),
        padding='same',
        use_bias=False,
        name='%s_Conv1' % model_name)(inputs)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3, momentum=0.999, name='%s_bn_Conv1' % model_name)(x)
    x = layers.ReLU(6., name='%s_Conv1_relu' % model_name)(x)

    x = _inverted_res_block(model_name,
                            x, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0)

    x = _inverted_res_block(model_name,
                            x, filters=24, alpha=alpha, stride=2, expansion=6, block_id=1)
    x = _inverted_res_block(model_name,
                            x, filters=24, alpha=alpha, stride=1, expansion=6, block_id=2)

    x = _inverted_res_block(model_name,
                            x, filters=32, alpha=alpha, stride=2, expansion=6, block_id=3)
    x = _inverted_res_block(model_name,
                            x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=4)
    x = _inverted_res_block(model_name,
                            x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=5)

    x = _inverted_res_block(model_name,
                            x, filters=64, alpha=alpha, stride=2, expansion=6, block_id=6)
    x = _inverted_res_block(model_name,
                            x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=7)
    x = _inverted_res_block(model_name,
                            x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=8)
    x = _inverted_res_block(model_name,
                            x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=9)

    x = _inverted_res_block(model_name,
                            x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=10)
    x = _inverted_res_block(model_name,
                            x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=11)
    x = _inverted_res_block(model_name,
                            x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=12)

    x = _inverted_res_block(model_name,
                            x, filters=160, alpha=alpha, stride=2, expansion=6, block_id=13)
    x = _inverted_res_block(model_name,
                            x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=14)
    x = _inverted_res_block(model_name,
                            x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=15)

    x = _inverted_res_block(model_name,
                            x, filters=320, alpha=alpha, stride=1, expansion=6, block_id=16)

    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    x = layers.Conv2D(
        last_block_filters, kernel_size=1, use_bias=False, name='%s_Conv_1' % model_name)(x)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3, momentum=0.999, name='%s_Conv_1_bn' % model_name)(x)
    x = layers.ReLU(6., name='%s_out_relu' % model_name)(x)

    if include_top:
        prediction_layer = PredictionLayer(num_classes=classes)
        x = prediction_layer(x)

    model = training.Model(inputs, x)

    return model


def create_mobilenet_v2(input_shape=None,
                        alpha=1.0,
                        include_top=True,
                        weights='imagenet',
                        input_tensor=None,
                        pooling=None,
                        classes=1000,
                        classifier_activation='softmax',
                        **kwargs):
    # if kwargs:
    #     raise ValueError('Unknown argument(s): %s' % (kwargs,))
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top` '
                         'as true, `classes` should be 1000')

    # Determine proper input shape and default size.
    # If both input_shape and input_tensor are used, they should match
    if input_shape is not None and input_tensor is not None:
        try:
            is_input_t_tensor = backend.is_keras_tensor(input_tensor)
        except ValueError:
            try:
                is_input_t_tensor = backend.is_keras_tensor(
                    layer_utils.get_source_inputs(input_tensor))
            except ValueError:
                raise ValueError('input_tensor: ', input_tensor,
                                 'is not type input_tensor')
        if is_input_t_tensor:
            if backend.image_data_format == 'channels_first':
                if backend.int_shape(input_tensor)[1] != input_shape[1]:
                    raise ValueError('input_shape: ', input_shape, 'and input_tensor: ',
                                     input_tensor,
                                     'do not meet the same shape requirements')
            else:
                if backend.int_shape(input_tensor)[2] != input_shape[1]:
                    raise ValueError('input_shape: ', input_shape, 'and input_tensor: ',
                                     input_tensor,
                                     'do not meet the same shape requirements')
        else:
            raise ValueError('input_tensor specified: ', input_tensor,
                             'is not a keras tensor')

    default_size = 224

    # If input_shape is None, infer shape from input_tensor
    if input_shape is None and input_tensor is not None:

        try:
            backend.is_keras_tensor(input_tensor)
        except ValueError:
            raise ValueError('input_tensor: ', input_tensor, 'is type: ',
                             type(input_tensor), 'which is not a valid type')

        if input_shape is None and not backend.is_keras_tensor(input_tensor):
            default_size = 224
        elif input_shape is None and backend.is_keras_tensor(input_tensor):
            if backend.image_data_format() == 'channels_first':
                rows = backend.int_shape(input_tensor)[2]
                cols = backend.int_shape(input_tensor)[3]
            else:
                rows = backend.int_shape(input_tensor)[1]
                cols = backend.int_shape(input_tensor)[2]

            if rows == cols and rows in [96, 128, 160, 192, 224]:
                default_size = rows
            else:
                default_size = 224

    # If input_shape is None and no input_tensor
    elif input_shape is None:
        default_size = 224

    # If input_shape is not None, assume default size
    else:
        if backend.image_data_format() == 'channels_first':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            rows = input_shape[0]
            cols = input_shape[1]

        if rows == cols and rows in [96, 128, 160, 192, 224]:
            default_size = rows

    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=default_size,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights)

    if backend.image_data_format() == 'channels_last':
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    if weights == 'imagenet':
        if alpha not in [0.35, 0.50, 0.75, 1.0, 1.3, 1.4]:
            raise ValueError('If imagenet weights are being loaded, '
                             'alpha can be one of `0.35`, `0.50`, `0.75`, '
                             '`1.0`, `1.3` or `1.4` only.')

        if rows != cols or rows not in [96, 128, 160, 192, 224]:
            rows = 224
            logging.warning('`input_shape` is undefined or non-square, '
                            'or `rows` is not in [96, 128, 160, 192, 224].'
                            ' Weights for input shape (224, 224) will be'
                            ' loaded as the default.')

    if 'sub_name' in kwargs:
        sub_name = kwargs.pop('sub_name')
        model_name = 'mobilenetv2_%s_%0.2f_%s' % (sub_name, alpha, rows)
    else:
        model_name = 'mobilenetv2_%0.2f_%s' % (alpha, rows)

    if input_tensor is None:
        img_input = layers.Input(name='%s_input' % model_name, shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    first_block_filters = _make_divisible(32 * alpha, 8)
    x = layers.ZeroPadding2D(
        padding=imagenet_utils.correct_pad(img_input, 3),
        name='%s_Conv1_pad' % model_name)(img_input)
    x = layers.Conv2D(
        first_block_filters,
        kernel_size=3,
        strides=(2, 2),
        padding='valid',
        use_bias=False,
        name='%s_Conv1' % model_name)(x)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3, momentum=0.999, name='%s_bn_Conv1' % model_name)(x)
    x = layers.ReLU(6., name='%s_Conv1_relu' % model_name)(x)

    x = _inverted_res_block(model_name,
                            x, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0)

    x = _inverted_res_block(model_name,
                            x, filters=24, alpha=alpha, stride=2, expansion=6, block_id=1)
    x = _inverted_res_block(model_name,
                            x, filters=24, alpha=alpha, stride=1, expansion=6, block_id=2)

    x = _inverted_res_block(model_name,
                            x, filters=32, alpha=alpha, stride=2, expansion=6, block_id=3)
    x = _inverted_res_block(model_name,
                            x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=4)
    x = _inverted_res_block(model_name,
                            x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=5)

    x = _inverted_res_block(model_name,
                            x, filters=64, alpha=alpha, stride=2, expansion=6, block_id=6)
    x = _inverted_res_block(model_name,
                            x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=7)
    x = _inverted_res_block(model_name,
                            x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=8)
    x = _inverted_res_block(model_name,
                            x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=9)

    x = _inverted_res_block(model_name,
                            x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=10)
    x = _inverted_res_block(model_name,
                            x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=11)
    x = _inverted_res_block(model_name,
                            x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=12)

    x = _inverted_res_block(model_name,
                            x, filters=160, alpha=alpha, stride=2, expansion=6, block_id=13)
    x = _inverted_res_block(model_name,
                            x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=14)
    x = _inverted_res_block(model_name,
                            x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=15)

    x = _inverted_res_block(model_name,
                            x, filters=320, alpha=alpha, stride=1, expansion=6, block_id=16)

    # no alpha applied to last conv as stated in the paper:
    # if the width multiplier is greater than 1 we
    # increase the number of output channels
    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    x = layers.Conv2D(
        last_block_filters, kernel_size=1, use_bias=False, name='%s_Conv_1' % model_name)(x)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3, momentum=0.999, name='%s_Conv_1_bn' % model_name)(x)
    x = layers.ReLU(6., name='%s_out_relu' % model_name)(x)

    if include_top:
        x = layers.GlobalAveragePooling2D()(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Dense(classes, activation=classifier_activation,
                         name='%s_predictions' % model_name)(x)

    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = training.Model(inputs, x, name=model_name)

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            model_name = ('mobilenet_v2_weights_tf_dim_ordering_tf_kernels_' +
                          str(alpha) + '_' + str(rows) + '.h5')
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = data_utils.get_file(
                model_name, weight_path, cache_subdir='models')
        else:
            model_name = ('mobilenet_v2_weights_tf_dim_ordering_tf_kernels_' +
                          str(alpha) + '_' + str(rows) + '_no_top' + '.h5')
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = data_utils.get_file(
                model_name, weight_path, cache_subdir='models')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model
