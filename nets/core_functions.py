from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from nets.const_vars import IMG_SIZE_224


def global_pool(shape, pool_op=keras.layers.AvgPool2D):
    pool_size = [shape[1], shape[2]]
    output = pool_op(pool_size=pool_size, strides=[1, 1], padding="valid")
    return output


def load_weight(var_loaded, all_vars, show_model_weights=False):
    result = False
    if var_loaded:
        var_loaded_fixed_name = {}
        for key in var_loaded:
            var_loaded_fixed_name.update({key + ":0": var_loaded[key]})

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
    return result


def load_weight_from_old_checkpoint(latest_checkpoint, target_model, model_name, **kwargs):
    value_to_load = kwargs.get("value_to_load", {})
    key_to_numpy = kwargs.get("key_to_numpy", {})
    pop_key = kwargs.get("pop_key", True)
    include_prediction_layer = kwargs.get("include_prediction_layer", True)

    if not bool(key_to_numpy):
        reader = tf.compat.v1.train.NewCheckpointReader(latest_checkpoint)
        var_to_shape_map = reader.get_variable_to_shape_map()
        value_to_load = {}
        key_to_numpy = {}
        for key in sorted(var_to_shape_map.items()):
            key_to_numpy.update({key[0]: reader.get_tensor(key[0])})

    key_maps1 = {
        "Conv1/kernel": "Conv/weights",
        "bn_Conv1/gamma": "Conv/BatchNorm/gamma",
        "bn_Conv1/beta": "Conv/BatchNorm/beta",
        "bn_Conv1/moving_mean": "Conv/BatchNorm/moving_mean",
        "bn_Conv1/moving_variance": "Conv/BatchNorm/moving_variance",
        "expanded_conv_depthwise/depthwise_kernel": "expanded_conv/depthwise/depthwise_weights",
        "expanded_conv_depthwise_BN/gamma": "expanded_conv/depthwise/BatchNorm/gamma",
        "expanded_conv_depthwise_BN/beta": "expanded_conv/depthwise/BatchNorm/beta",
        "expanded_conv_depthwise_BN/moving_mean": "expanded_conv/depthwise/BatchNorm/moving_mean",
        "expanded_conv_depthwise_BN/moving_variance": "expanded_conv/depthwise/BatchNorm/moving_variance",
        "expanded_conv_project/kernel": "expanded_conv/project/weights",
        "expanded_conv_project_BN/gamma": "expanded_conv/project/BatchNorm/gamma",
        "expanded_conv_project_BN/beta": "expanded_conv/project/BatchNorm/beta",
        "expanded_conv_project_BN/moving_mean": "expanded_conv/project/BatchNorm/moving_mean",
        "expanded_conv_project_BN/moving_variance": "expanded_conv/project/BatchNorm/moving_variance",
        "Conv_1/kernel": "Conv_1/weights",
        "Conv_1_bn/gamma": "Conv_1/BatchNorm/gamma",
        "Conv_1_bn/beta": "Conv_1/BatchNorm/beta",
        "Conv_1_bn/moving_mean": "Conv_1/BatchNorm/moving_mean",
        "Conv_1_bn/moving_variance": "Conv_1/BatchNorm/moving_variance",
    }
    key_maps2 = {
        "prediction_layer/prediction_layer/kernel": "Logits/Conv2d_1c_1x1/weights",
        "prediction_layer/prediction_layer/bias": "Logits/Conv2d_1c_1x1/biases",
    }
    key_maps3 = {
        "block_{}_expand/kernel": "expanded_conv_{}/expand/weights",
        "block_{}_expand_BN/gamma": "expanded_conv_{}/expand/BatchNorm/gamma",
        "block_{}_expand_BN/beta": "expanded_conv_{}/expand/BatchNorm/beta",
        "block_{}_expand_BN/moving_mean": "expanded_conv_{}/expand/BatchNorm/moving_mean",
        "block_{}_expand_BN/moving_variance": "expanded_conv_{}/expand/BatchNorm/moving_variance",
        "block_{}_depthwise/depthwise_kernel": "expanded_conv_{}/depthwise/depthwise_weights",
        "block_{}_depthwise_BN/gamma": "expanded_conv_{}/depthwise/BatchNorm/gamma",
        "block_{}_depthwise_BN/beta": "expanded_conv_{}/depthwise/BatchNorm/beta",
        "block_{}_depthwise_BN/moving_mean": "expanded_conv_{}/depthwise/BatchNorm/moving_mean",
        "block_{}_depthwise_BN/moving_variance": "expanded_conv_{}/depthwise/BatchNorm/moving_variance",
        "block_{}_project/kernel": "expanded_conv_{}/project/weights",
        "block_{}_project_BN/gamma": "expanded_conv_{}/project/BatchNorm/gamma",
        "block_{}_project_BN/beta": "expanded_conv_{}/project/BatchNorm/beta",
        "block_{}_project_BN/moving_mean": "expanded_conv_{}/project/BatchNorm/moving_mean",
        "block_{}_project_BN/moving_variance": "expanded_conv_{}/project/BatchNorm/moving_variance",
    }

    for key in key_maps1:
        _key = model_name + "/" + key_maps1[key]
        if _key in key_to_numpy:
            value = key_to_numpy[_key]
            value_to_load[target_model + key] = value
            if pop_key:
                key_to_numpy.pop(_key)
        else:
            print("Can't find the key: {}".format(_key))

    if include_prediction_layer:
        for key in key_maps2:
            _key = model_name + "/" + key_maps2[key]
            if _key in key_to_numpy:
                value = key_to_numpy[_key]
                value_to_load[key] = value
                if pop_key:
                    key_to_numpy.pop(_key)
            else:
                print("Can't find the key: {}".format(_key))

    for i in range(1, 17):
        for key in key_maps3:
            k = model_name + "/" + key_maps3[key]
            _key_v = k.format(i)
            if _key_v in key_to_numpy:
                value = key_to_numpy[_key_v]
                value_to_load[target_model + key.format(i)] = value
                if pop_key:
                    key_to_numpy.pop(_key_v)
            else:
                print("Can't find the key: {}".format(_key_v))
    return value_to_load


def load_orchids52_weight_from_old_checkpoint(latest_checkpoint):
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

    return key_to_numpy, var_maps


@tf.function
def preprocess_input(image_data, central_fraction=0.875):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.central_crop(image, central_fraction=central_fraction)
    image = tf.image.resize(images=image, size=IMG_SIZE_224, method=tf.image.ResizeMethod.BILINEAR)

    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)

    return image
