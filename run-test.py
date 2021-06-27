from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from lib_utils import start
from lib_utils import FLAGS
from lib_utils import create_image_lists
from lib_utils import DisplayInfo
from nets.utils import nets_mapping
from nets.utils import preprocessing_mapping
from data import data_utils
from absl import logging


def main(unused_argv):
    logging.debug(unused_argv)
    dataset_images = create_image_lists(image_dir=FLAGS.image_dir)
    load_dataset = data_utils.dataset_mapping[FLAGS.dataset]
    create_model = nets_mapping[FLAGS.model]
    preprocess_input = preprocessing_mapping[FLAGS.model]

    model = create_model(num_classes=load_dataset.num_of_classes, optimizer=None, loss_fn=None, batch_size=1, step="")
    accuracy_metric = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")
    model.compile(metrics=[accuracy_metric])
    model.restore_model_variables(checkpoint_path=FLAGS.checkpoint_path)
    model.summary()

    info = DisplayInfo(load_dataset.test_size)

    count = 0
    for label, data in dataset_images.items():
        for file in data["testing"]:
            filename = os.path.join(FLAGS.image_dir, data["dir"], file)
            image_data = tf.io.gfile.GFile(filename, "rb").read()
            inputs = preprocess_input(image_data)
            result = model.model(inputs).numpy()

            count = count + 1
            info.display_info(result, label, count)

    info.display_summary()


if __name__ == "__main__":
    #tf.config.run_functions_eagerly(True)
    start(main)
