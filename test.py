from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from data import data_utils
from data.data_utils import dataset_mapping
from lib_utils import start
logging = tf.compat.v1.logging

def printname(name):
    print(name)


def main(unused_argv):
    logging.debug(unused_argv)
    data_path = os.environ['DATA_DIR'] or '/Volumes/Data/_dataset/_orchids_dataset'
    data_dir = os.path.join(data_path, 'orchids52_data')
    load_dataset = dataset_mapping[data_utils.ORCHIDS52_V1_TFRECORD]
    train_ds = load_dataset(split="train", batch_size=2, root_path=data_dir)

    img, lbl = next(iter(train_ds))
    print(lbl)


if __name__ == '__main__':
    tf.config.experimental_run_functions_eagerly(True)
    start(main)