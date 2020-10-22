from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from data import orchids52_dataset_file
from data import orchids52_dataset_tfrecord


MOBILENET_V2_FILE = 'const/MOBILENET_V2_FILE'
MOBILENET_V2_TFRECORD = 'const/MOBILENET_V2_TFRECORD'

logging = tf.compat.v1.logging


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


dataset_mapping = {
    MOBILENET_V2_FILE: orchids52_dataset_file,
    MOBILENET_V2_TFRECORD: orchids52_dataset_tfrecord
}
