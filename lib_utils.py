from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pathlib
import functools

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.platform import tf_logging as logging
from stn import pre_spatial_transformer_network

default_image_size = 224
IMG_SIZE_224 = (default_image_size, default_image_size)
IMG_SHAPE_224 = IMG_SIZE_224 + (3,)
BASE_WEIGHT_PATH = ('https://storage.googleapis.com/tensorflow/'
                    'keras-applications/mobilenet_v2/')


def wrapped_partial(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func


def get_label(file_path, class_names):
    parts = tf.strings.split(file_path, os.path.sep)
    one_hot = parts[-2] == class_names
    return tf.cast(one_hot, tf.float32)


def decode_img(img, size):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # resize the image to the desired size
    return tf.image.resize(img, size)


def process_path(file_path, class_names, image_size):
    label = get_label(file_path, class_names)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img, image_size)
    return img, label


def configure_for_performance(ds, batch_size=32):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


def get_step_number(checkpoint_dir):
    idx = checkpoint_dir.index('.')
    step = int(checkpoint_dir[:idx][-4:])
    return step


def latest_checkpoint(checkpoint_path):
    file_path = pathlib.Path(checkpoint_path)
    file_list = list(file_path.glob('*.h5'))

    if len(file_list) > 0:
        max_step = 0
        for file in file_list:
            step = get_step_number(file.name)
            if max_step < step:
                max_step = step
                file_path = file
        return file_path, max_step
    return None, 0


def create_dataset(batch_size):
    train_data_dir = pathlib.Path("/Volumes/Data/_dataset/_orchids_dataset/orchids52_data/train-en")
    test_data_dir = pathlib.Path("/Volumes/Data/_dataset/_orchids_dataset/orchids52_data/test-en")

    image_count = len(list(train_data_dir.glob('*/*.jpg')))
    train_ds = tf.data.Dataset.list_files(str(train_data_dir / '*/*'), shuffle=False)
    train_ds = train_ds.shuffle(image_count, reshuffle_each_iteration=False)
    test_ds = tf.data.Dataset.list_files(str(test_data_dir / '*/*'), shuffle=False)

    class_names = np.array(sorted([item.name for item in train_data_dir.glob('*')]))
    print(class_names)

    val_batches = tf.data.experimental.cardinality(test_ds)
    val_ds = test_ds.skip(val_batches // 5)
    test_ds = test_ds.take(val_batches // 5)

    print(tf.data.experimental.cardinality(train_ds).numpy())
    print(tf.data.experimental.cardinality(val_ds).numpy())
    print(tf.data.experimental.cardinality(test_ds).numpy())

    _process_path = wrapped_partial(
        process_path,
        class_names=class_names,
        image_size=IMG_SIZE_224)

    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    train_ds = train_ds.map(_process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(_process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.map(_process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_ds = configure_for_performance(train_ds, batch_size=batch_size)
    val_ds = configure_for_performance(val_ds, batch_size=batch_size)
    test_ds = configure_for_performance(test_ds, batch_size=batch_size)

    num_classes = len(class_names)

    return train_ds, val_ds, test_ds, num_classes


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


def create_mobilenet_v2(input_shape=None,
                        alpha=1.0,
                        include_top=True,
                        weights='imagenet',
                        input_tensor=None,
                        pooling=None,
                        classes=1000,
                        classifier_activation='softmax',
                        **kwargs):
    if 'layers' in kwargs:
        global layers
        layers = kwargs.pop('layers')

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
        else:
            default_size = 224

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



def create_orchid_mobilenet_v2_14(num_classes, freeze_base_model=False):
    # Create the base model from the pre-trained model MobileNet V2
    base_model = create_mobilenet_v2(input_shape=IMG_SHAPE_224,
                                     alpha=1.4,
                                     include_top=False,
                                     weights='imagenet')

    data_augmentation = keras.Sequential([
        keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    preprocess_input = keras.applications.mobilenet_v2.preprocess_input
    global_average_layer = keras.layers.GlobalAveragePooling2D()
    prediction_layer = keras.layers.Dense(num_classes)

    inputs = keras.Input(shape=IMG_SHAPE_224)
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = keras.Model(inputs, outputs)

    if freeze_base_model:
        # Freeze all the layers except for dense layer
        for layer in base_model.layers:
            layer.trainable = False

    model.summary()

    return model


def create_orchid_mobilenet_v2_14_cus(num_classes,
                                      freeze_base_model=False,
                                      is_training=False,
                                      **kwargs):
    ds = kwargs.pop('ds')
    batch, _ = next(iter(ds))

    global_average_layer = keras.layers.GlobalAveragePooling2D()

    data_augmentation = keras.Sequential([
        keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    preprocess_input = keras.applications.mobilenet_v2.preprocess_input

    scales = [0.8, 0.6]

    inputs = keras.Input(batch_input_shape=batch.shape)
    inputs = data_augmentation(inputs)
    inputs = preprocess_input(inputs)

    # Create the base model from the pre-trained model MobileNet V2
    base_model1 = create_mobilenet_v2(input_tensor=inputs,
                                      alpha=1.4,
                                      include_top=False,
                                      weights='imagenet',
                                      sub_name='01')
    base_model2 = create_mobilenet_v2(input_shape=IMG_SHAPE_224,
                                      alpha=1.4,
                                      include_top=False,
                                      weights='imagenet',
                                      sub_name='02')

    x = base_model1(inputs, training=False)
    x = layers.Conv2D(128, [1, 1], activation='relu', name="conv2d_resize_128")(x)
    x = layers.Flatten()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)

    element_size = 3  # [x, y, scale]
    fc_num = element_size * len(scales)
    h_fc1 = layers.Dense(fc_num, activation='tanh', activity_regularizer='l2')(x)

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

    all_logits = []
    x = base_model2(inputs, training=False)
    all_logits.append(x)
    for i, input_image in enumerate(all_images):
        x = base_model2(input_image, training=False)
        all_logits.append(x)

    all_predicts = []
    for i, net in enumerate(all_logits):
        x = global_average_layer(net)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(num_classes)(x)
        all_predicts.append(x)

    if 'step' in kwargs:
        step = kwargs.pop('step')
    else:
        step = ''

    if step == 'pretrain1':
        outputs = tf.add_n(all_predicts)
        outputs = tf.divide(outputs, len(all_predicts))
    else:
        c_t = None
        main_net = None
        for i, net in enumerate(all_predicts):
            if i > 0:
                input_and_hstate_concatenated = tf.concat(axis=1, values=[c_t, net])
                c_t = layers.Dense(num_classes,
                                   kernel_initializer=initializers.truncated_normal(mean=0.5)
                                   )(input_and_hstate_concatenated)
                main_net = main_net + c_t
            else:
                main_net = net
                c_t = net
        outputs = main_net
    model = keras.Model(inputs, outputs)

    if freeze_base_model:
        # Freeze all the layers except for dense layer
        for layer in base_model1.layers:
            layer.trainable = False
        for layer in base_model2.layers:
            layer.trainable = False

    model.summary()

    return model