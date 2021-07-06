from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

from absl import logging
from data import data_utils
from utils.lib_utils import create_image_lists
from utils.lib_utils import FLAGS
from utils.lib_utils import DisplayInfo
from utils.lib_utils import start
from nets.mobilenet_v2_140 import preprocess_input


def main(unused_argv):
    logging.debug(unused_argv)
    dataset_images = create_image_lists(image_dir=FLAGS.image_dir)
    load_dataset = data_utils.dataset_mapping[FLAGS.dataset]
    model = tf.keras.models.load_model(FLAGS.checkpoint_path)
    model.summary()

    info = DisplayInfo(load_dataset.test_size)

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
    start(main)
