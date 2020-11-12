from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from data import orchids52_dataset_file
from data import orchids52_dataset_tfrecord


ORCHIDS52_V1_FILE = 'orchids52_v1_file'
ORCHIDS52_V1_TFRECORD = 'orchids52_v1_tf'
ORCHIDS52_V2_FILE = 'orchids52_v2_file'
ORCHIDS52_V2_TFRECORD = 'orchids52_v2_tf'

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


def get_data_files(data_sources):
  if isinstance(data_sources, (list, tuple)):
    data_files = []
    for source in data_sources:
      data_files += get_data_files(source)
  else:
    if '*' in data_sources or '?' in data_sources or '[' in data_sources:
      data_files = tf.io.gfile.glob(data_sources)
    else:
      data_files = [data_sources]
  if not data_files:
    raise ValueError('No data files found in %s' % (data_sources,))
  return data_files


dataset_mapping = {
    ORCHIDS52_V1_FILE: orchids52_dataset_file.load_dataset_v1,
    ORCHIDS52_V1_TFRECORD: orchids52_dataset_tfrecord.load_dataset_v1,
    ORCHIDS52_V2_FILE: orchids52_dataset_file.load_dataset_v2,
    ORCHIDS52_V2_TFRECORD: orchids52_dataset_tfrecord.load_dataset_v2
}
