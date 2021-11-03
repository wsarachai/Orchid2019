from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from nets.preprocessing import preprocess_image

import os
import tensorflow as tf

from utils.lib_utils import DisplayInfo
from nets.mapping import nets_mapping
from data import data_utils
from absl import logging
from utils.start_app import FLAGS, start


def main(unused_argv):
    logging.debug(unused_argv)

    # tf.config.run_functions_eagerly(True)

    workspace_path = os.environ["WORKSPACE"] if "WORKSPACE" in os.environ else "/Users/watcharinsarachai/Documents/"
    checkpoint_dir = os.path.join(workspace_path, "_trained_models", "orchids2019", FLAGS.checkpoint_dir)

    datasets = data_utils.load_dataset(flags=FLAGS, workspace_path=workspace_path, split="test", preprocessing=True)

    if FLAGS.dataset_format == "tf-records":
        size = datasets.size
        num_of_classes = datasets.num_of_classes

        datasets = datasets.map(
            lambda img, lbl: (preprocess_image(img, width=224, height=224, is_training=False), lbl)
        )

        datasets = datasets.batch(batch_size=FLAGS.batch_size)
        datasets = datasets.cache().prefetch(buffer_size=AUTOTUNE)

        datasets.size = size
        datasets.num_of_classes = num_of_classes

    create_model = nets_mapping[FLAGS.model]

    model = create_model(
        num_classes=datasets.num_of_classes, activation="softmax", step=FLAGS.train_step, training=False
    )
    accuracy_metric = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")
    model.compile(metrics=[accuracy_metric])
    model.restore_model_variables(checkpoint_dir=checkpoint_dir)
    model.summary()

    info = DisplayInfo(datasets.size, training_step=FLAGS.train_step)

    @tf.function
    def process_step(model, inputs):
        return model(inputs)

    count = 0
    for inputs, label in datasets:
        result = process_step(model.model, inputs).numpy()
        count = count + 1
        info.display_info(result, label, count)

    info.display_summary()

    # checkpoint_dir = os.path.join(workspace_path, "_trained_models", "orchids2019", FLAGS.model)
    # model.save(checkpoint_dir + ".h5")


if __name__ == "__main__":
    start(main)
