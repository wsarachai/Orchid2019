from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from absl import logging
from data.data_utils import load_dataset
from utils.lib_utils import FLAGS
from utils.lib_utils import DisplayInfo
from utils.lib_utils import start
from nets.mapping import nets_mapping
from utils.lib_utils import config_optimizer
from utils import const


def main(unused_argv):
    logging.debug(unused_argv)

    split = "test"
    workspace_path = os.environ["WORKSPACE"] if "WORKSPACE" in os.environ else "/Users/watcharinsarachai/Documents/"
    checkpoint_dir = os.path.join(workspace_path, FLAGS.checkpoint_dir, const.TRAIN_TEMPLATE.format(FLAGS.train_step))

    test_ds = load_dataset(flags=FLAGS, workspace_path=workspace_path, split="test", preprocessing=True)

    create_model = nets_mapping[FLAGS.model]
    model = create_model(num_classes=test_ds.num_of_classes, step=FLAGS.train_step, batch_size=FLAGS.batch_size)

    optimizer = config_optimizer(FLAGS.optimizer, FLAGS.learning_rate)

    model_checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model.model)
    checkpoint.restore(model_checkpoint_path)
    model.summary()

    info = DisplayInfo(test_ds.size)

    count = 0
    for data, label in test_ds:
        result = model.model(data).numpy()

        count += 1
        info.display_info(result, label[0], count)

    info.display_summary()


if __name__ == "__main__":
    # tf.config.run_functions_eagerly(True)
    start(main)
