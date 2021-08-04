from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from absl import logging
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from nets.mapping import nets_mapping
from data.data_utils import load_dataset
from utils.lib_utils import config_learning_rate
from utils.lib_utils import config_optimizer
from utils.lib_utils import config_loss
from utils.start_app import FLAGS, start
from utils.training_utils import TrainClassifier
from utils import const
from nets.preprocessing import preprocess_image


def main(unused_argv):
    logging.debug(unused_argv)

    workspace_path = os.environ["WORKSPACE"] if "WORKSPACE" in os.environ else "/Users/watcharinsarachai/Documents/"
    create_model = nets_mapping[FLAGS.model]

    trained_weights_dir = None if FLAGS.trained_dir is None else os.path.join(workspace_path, FLAGS.trained_dir)
    training_dir = os.path.join(workspace_path, "_trained_models", "orchids2019", FLAGS.model)

    test_ds = load_dataset(flags=FLAGS, workspace_path=workspace_path, split="test", preprocessing=True, one_hot=True)

    if FLAGS.dataset_format == "tf-records":
        test_size = test_ds.size
        num_of_classes = test_ds.num_of_classes
        test_ds = test_ds.map(lambda img, lbl: (preprocess_image(img, width=224, height=224, is_training=False), lbl))
        test_ds = test_ds.batch(batch_size=FLAGS.batch_size)
        test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

        test_ds.size = test_size
        test_ds.num_of_classes = num_of_classes

    optimizer = config_optimizer(FLAGS.optimizer, learning_rate=FLAGS.learning_rate)
    loss_fn = config_loss()

    model = create_model(
        num_classes=test_ds.num_of_classes,
        optimizer=optimizer,
        loss_fn=loss_fn,
        training=False,
        step=FLAGS.train_step,
        activation="softmax",
        batch_size=FLAGS.batch_size,
        dropout=FLAGS.dropout,
    )

    log_dir = os.path.join(training_dir, "logs", const.TRAIN_TEMPLATE.format(FLAGS.train_step))

    train_model = TrainClassifier(
        model=model,
        batch_size=FLAGS.batch_size,
        summary_path=log_dir,
        epoches=FLAGS.total_epochs,
        data_handler_steps=None,
        test_ds=test_ds,
        moving_average_decay=None,
        callbacks=[],
    )

    model.config_checkpoint(training_dir)
    model.restore_model_variables(
        checkpoint_dir=training_dir,
        training_for_tf25=True,
        pop_key=False,
        training_step=FLAGS.train_step,
        # average_vars=average_vars,
    )

    model.summary()
    train_model.reset_metric()
    train_model.evaluate(datasets=test_ds)


if __name__ == "__main__":
    # tf.config.run_functions_eagerly(True)
    start(main)
