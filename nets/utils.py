from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras import Sequential
from nets.mobilenet_v2_140 import create_mobilenet_v2_14
from nets.mobilenet_v2_140_orchids52 import create_orchid_mobilenet_v2_14

TRAIN_TEMPLATE = 'pretrain{step}'
TRAIN_STEP1 = 'pretrain1'
TRAIN_STEP2 = 'pretrain2'
TRAIN_STEP3 = 'pretrain3'
TRAIN_STEP4 = 'pretrain4'
TRAIN_V2_STEP1 = 'v2-pretrain1'
TRAIN_V2_STEP2 = 'v2-pretrain2'

MOBILENET_V2_140 = 'mobilenet_v2_140'
MOBILENET_V2_140_ORCHIDS52 = 'mobilenet_v2_140_orchids52'


nets_mapping = {
    MOBILENET_V2_140: create_mobilenet_v2_14,
    MOBILENET_V2_140_ORCHIDS52: create_orchid_mobilenet_v2_14
}


def create_predict_module(num_classes, name, activation='linear'):
    module = Sequential([
        keras.layers.GlobalAveragePooling2D(name='{}_global'.format(name)),
        keras.layers.Dropout(0.2, name='{}_droupout'.format(name)),
        keras.layers.Dense(num_classes, activation=activation, name='{}_fc'.format(name))
    ], name='{}_prediction_layer'.format(name))
    return module
