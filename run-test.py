from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from utils.lib_utils import start
from utils.lib_utils import FLAGS
from utils.lib_utils import DisplayInfo
from nets.mapping import nets_mapping
from data import data_utils
from absl import logging


def main(unused_argv):
    logging.debug(unused_argv)

    workspace_path = os.environ["WORKSPACE"] if "WORKSPACE" in os.environ else "/Users/watcharinsarachai/Documents/"
    checkpoint_dir = os.path.join(workspace_path, "_trained_models", "model-v1", FLAGS.checkpoint_dir)

    datasets = data_utils.load_dataset(flags=FLAGS, workspace_path=workspace_path, split="test", preprocessing=True)

    create_model = nets_mapping[FLAGS.model]

    model = create_model(num_classes=datasets.num_of_classes, activation="softmax")
    accuracy_metric = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")
    model.compile(metrics=[accuracy_metric])
    model.restore_model_variables(checkpoint_path=checkpoint_dir)
    model.summary()

    info = DisplayInfo(datasets.size)

    count = 0
    for inputs, label in datasets:
        result = model.model(inputs).numpy()
        count = count + 1
        info.display_info(result, label, count)

    info.display_summary()

    checkpoint_dir = os.path.join(workspace_path, "_trained_models", "orchids2019", FLAGS.model)
    model.save(checkpoint_dir)


if __name__ == "__main__":
    # tf.config.run_functions_eagerly(True)
    start(main)
