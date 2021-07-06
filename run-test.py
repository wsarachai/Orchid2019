from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import h5py
import numpy as np
import tensorflow as tf

from utils.lib_utils import start
from utils.lib_utils import FLAGS
from utils.lib_utils import DisplayInfo
from nets.mapping import nets_mapping
from nets.mapping import preprocessing_mapping
from data import data_utils
from absl import logging


def main(unused_argv):
    logging.debug(unused_argv)

    workspace_path = os.environ["WORKSPACE"] if "WORKSPACE" in os.environ else "/Users/watcharinsarachai/Documents/"
    checkpoint_path = os.path.join(
        workspace_path, "_trained_models", "orchids2019", FLAGS.checkpoint_path, "variables", "variables"
    )

    datasets = data_utils.load_dataset(flags=FLAGS, workspace_path=workspace_path, split="test")
    create_model = nets_mapping[FLAGS.model]
    preprocess_input = preprocessing_mapping[FLAGS.model]

    model = create_model(num_classes=datasets.num_of_classes, activation="softmax")
    accuracy_metric = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")
    model.compile(metrics=[accuracy_metric])
    model.load_weights(checkpoint_path=checkpoint_path)
    model.summary()

    info = DisplayInfo(datasets.size)

    count = 0
    for inputs, label in datasets:
        if FLAGS.dataset_format == "files":
            inputs = preprocess_input(inputs)
        result = model.model(inputs).numpy()
        count = count + 1
        info.display_info(result, label, count)

    info.display_summary()


if __name__ == "__main__":
    # tf.config.run_functions_eagerly(True)
    start(main)
