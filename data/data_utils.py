from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from data import orchids52_dataset_file
from data import flowers17_dataset_v1_tfrecord
from data import flowers102_dataset_v1_tfrecord
from data import orchids52_dataset_v1_tfrecord
from data import orchids52_dataset_tfrecord
from data import orchids52_dataset_h5


FLOWERS17 = "flowers17_data"
FLOWERS102 = "flowers102_data"
ORCHIDS52 = "orchids52_data"

DATA_FORMAT_TF_RECORDS = "tf-records"
DATA_FORMAT_FILES = "files"
DATA_FORMAT_H5 = "h5"

DATASET_VERSION_V1 = "v1"
DATASET_VERSION_V2 = "v2"

PATTERNS = "{}-{}-{}"
ORCHIDS52_V1_FILE = PATTERNS.format(ORCHIDS52, DATA_FORMAT_FILES, DATASET_VERSION_V1)
ORCHIDS52_V2_FILE = PATTERNS.format(ORCHIDS52, DATA_FORMAT_FILES, DATASET_VERSION_V2)
ORCHIDS52_V1_TFRECORD = PATTERNS.format(ORCHIDS52, DATA_FORMAT_TF_RECORDS, DATASET_VERSION_V1)
ORCHIDS52_V2_TFRECORD = PATTERNS.format(ORCHIDS52, DATA_FORMAT_TF_RECORDS, DATASET_VERSION_V2)
ORCHIDS52_V1_H5 = PATTERNS.format(ORCHIDS52, DATA_FORMAT_H5, DATASET_VERSION_V1)
ORCHIDS52_V2_H5 = PATTERNS.format(ORCHIDS52, DATA_FORMAT_H5, DATASET_VERSION_V2)
FLOWERS17_V1_FILE = PATTERNS.format(FLOWERS17, DATA_FORMAT_FILES, DATASET_VERSION_V1)
FLOWERS102_V1_FILE = PATTERNS.format(FLOWERS102, DATA_FORMAT_FILES, DATASET_VERSION_V1)
FLOWERS17_V1_TFRECORD = PATTERNS.format(FLOWERS17, DATA_FORMAT_TF_RECORDS, DATASET_VERSION_V1)
FLOWERS102_V1_TFRECORD = PATTERNS.format(FLOWERS102, DATA_FORMAT_TF_RECORDS, DATASET_VERSION_V1)
FLOWERS17_V1_H5 = PATTERNS.format(FLOWERS17, DATA_FORMAT_H5, DATASET_VERSION_V1)
FLOWERS102_V1_H5 = PATTERNS.format(FLOWERS102, DATA_FORMAT_H5, DATASET_VERSION_V2)


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def get_data_files(data_sources):
    if isinstance(data_sources, (list, tuple)):
        data_files = []
        for source in data_sources:
            data_files += get_data_files(source)
    else:
        if "*" in data_sources or "?" in data_sources or "[" in data_sources:
            data_files = tf.io.gfile.glob(data_sources)
        else:
            data_files = [data_sources]
    if not data_files:
        raise ValueError("No data files found in %s" % (data_sources,))
    return data_files


dataset_mapping = {
    ORCHIDS52_V1_FILE: orchids52_dataset_file.load_dataset_v1,
    ORCHIDS52_V2_FILE: orchids52_dataset_file.load_dataset_v2,
    FLOWERS17_V1_TFRECORD: flowers17_dataset_v1_tfrecord.load_dataset_v1,
    FLOWERS102_V1_TFRECORD: flowers102_dataset_v1_tfrecord.load_dataset_v1,
    ORCHIDS52_V1_TFRECORD: orchids52_dataset_v1_tfrecord.load_dataset_v1,
    ORCHIDS52_V2_TFRECORD: orchids52_dataset_tfrecord.load_dataset_v2,
    ORCHIDS52_V1_H5: orchids52_dataset_h5.load_dataset_v1,
    ORCHIDS52_V2_H5: orchids52_dataset_h5.load_dataset_v2,
    # FLOWERS17_V1_FILE: flowers17_dataset_file.load_dataset_v1,
    # FLOWERS102_V1_FILE: flowers102_dataset_file.load_dataset_v2,
    # FLOWERS17_V1_H5: flowers17_dataset_file.load_dataset_v1,
    # FLOWERS102_V1_H5: flowers102_dataset_file.load_dataset_v1,
}


def load_dataset(flags, workspace_path, split="train", **kwargs):
    dataset = PATTERNS.format(flags.dataset, flags.dataset_format, flags.dataset_version)
    data_dir = os.path.join(workspace_path, "_datasets", flags.dataset, flags.dataset_format, flags.dataset_version)
    return dataset_mapping[dataset](split=split, batch_size=flags.batch_size, root_path=data_dir, **kwargs)
