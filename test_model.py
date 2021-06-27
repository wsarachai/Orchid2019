from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import collections
import numpy as np
import tensorflow as tf
import lib_utils

from absl import logging
from data import data_utils
from lib_utils import create_image_lists
from lib_utils import FLAGS
from nets.mobilenet_v2_140 import preprocess_input


def main(unused_argv):
    logging.debug(unused_argv)
    dataset_images = create_image_lists(image_dir=FLAGS.image_dir)
    load_dataset = data_utils.dataset_mapping[FLAGS.dataset]
    model = tf.keras.models.load_model(FLAGS.checkpoint_path)
    model.summary()

    info = lib_utils.DisplayInfo(load_dataset.test_size)

    count = 0
    for label, data in dataset_images.items():
        for file in data["testing"]:
            filename = os.path.join(FLAGS.image_dir, data["dir"], file)
            image_data = tf.io.gfile.GFile(filename, "rb").read()
            inputs = preprocess_input(image_data)
            result = model(inputs).numpy()

            count += 1
            info.display_info(result, label, count)

    info.display_summary()


if __name__ == "__main__":
    # tf.config.run_functions_eagerly(True)
    lib_utils.start(main)
