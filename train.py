from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tensorflow as tf
from pickle import dump
from datetime import datetime
from absl import logging
from nets.mapping import nets_mapping
from data.data_utils import load_dataset
from utils.lib_utils import FLAGS
from utils.lib_utils import config_learning_rate
from utils.lib_utils import config_optimizer
from utils.lib_utils import config_loss
from utils.lib_utils import start
from utils import const


def main(unused_argv):
    logging.debug(unused_argv)

    # tf.config.run_functions_eagerly(True)

    workspace_path = os.environ["WORKSPACE"] if "WORKSPACE" in os.environ else "/Users/watcharinsarachai/Documents/"
    create_model = nets_mapping[FLAGS.model]

    trained_weights_dir = os.path.join(workspace_path, FLAGS.trained_dir)
    training_dir = os.path.join(workspace_path, "_trained_models", "orchids2019", FLAGS.model)
    if not tf.io.gfile.exists(training_dir):
        os.makedirs(training_dir)

    train_ds = load_dataset(
        flags=FLAGS, workspace_path=workspace_path, split="train", preprocessing=True, one_hot=True
    )
    test_ds = load_dataset(flags=FLAGS, workspace_path=workspace_path, split="test", preprocessing=True, one_hot=True)

    learning_rate_schedule = config_learning_rate(learning_rate=FLAGS.learning_rate, decay=FLAGS.learning_rate_decay)
    optimizer = config_optimizer(FLAGS.optimizer, learning_rate=FLAGS.learning_rate)
    loss_fn = config_loss()

    model = create_model(
        num_classes=train_ds.num_of_classes,
        optimizer=optimizer,
        loss_fn=loss_fn,
        training=True,
        step=FLAGS.train_step,
        activation="softmax",
        batch_size=FLAGS.batch_size,
    )

    checkpoint_dir = os.path.join(training_dir, const.TRAIN_TEMPLATE.format(FLAGS.train_step), "model")
    log_dir = os.path.join(training_dir, "logs", const.TRAIN_TEMPLATE.format(FLAGS.train_step))

    def scheduler(epochs, lr):
        return learning_rate_schedule(epochs)

    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,
                                                     save_weights_only=True,
                                                     verbose=1)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.config_checkpoint(checkpoint_dir)
    _checkpoint_dir = checkpoint_dir if FLAGS.train_step > 1 else trained_weights_dir
    epoch = model.restore_model_variables(
        checkpoint_dir=_checkpoint_dir, training_for_tf25=True, pop_key=False, training_step=FLAGS.train_step
    )
    model.summary()

    model.compile(metrics=['accuracy'])

    model.fit(train_ds, initial_epoch=epoch, epochs=FLAGS.total_epochs, validation_data=test_ds, callbacks=[callback, cp_callback, tensorboard_callback])

    print("\nTest accuracy: ")
    model.evaluate(datasets=test_ds)

    if FLAGS.save_model and model:
        model.save(checkpoint_dir)


def getParam(arg):
    if "=" in arg:
        vars = arg.split("=")
        return vars[1]
    raise ("Parameter format is invalid.")


if __name__ == "__main__":
    start(main)
