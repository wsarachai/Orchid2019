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
from utils.training_utils import TrainClassifier
from utils import const


def main(unused_argv):
    logging.debug(unused_argv)

    # tf.config.run_functions_eagerly(True)

    workspace_path = os.environ["WORKSPACE"] if "WORKSPACE" in os.environ else "/Users/watcharinsarachai/Documents/"
    create_model = nets_mapping[FLAGS.model]

    trained_dir = os.path.join(workspace_path, FLAGS.trained_dir)
    checkpoint_dir = os.path.join(workspace_path, "_trained_models", "orchids2019", FLAGS.model)
    if not tf.io.gfile.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    train_ds = load_dataset(
        flags=FLAGS, workspace_path=workspace_path, split="train", preprocessing=True, one_hot=True
    )
    test_ds = load_dataset(flags=FLAGS, workspace_path=workspace_path, split="test", preprocessing=True, one_hot=True)

    batch_size = FLAGS.batch_size

    # if FLAGS.train_step > 1:
    #     src_dir = os.path.join(checkpoint_dir, const.TRAIN_TEMPLATE.format(FLAGS.train_step - 1))
    #     des_dir = os.path.join(checkpoint_dir, const.TRAIN_TEMPLATE.format(FLAGS.train_step))
    #     if tf.io.gfile.exists(src_dir):
    #         if not tf.io.gfile.exists(des_dir):
    #             shutil.copytree(src_dir, des_dir)

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

    def scheduler(epoch, lr):
        return learning_rate_schedule(epoch)

    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    train_model = TrainClassifier(
        model=model,
        batch_size=batch_size,
        summary_path=os.path.join(checkpoint_dir, "logs", const.TRAIN_TEMPLATE.format(FLAGS.train_step)),
        epoches=FLAGS.total_epochs,
        data_handler_steps=train_ds,
        callbacks=[callback],
    )

    model.config_checkpoint(checkpoint_dir)
    _checkpoint_dir = checkpoint_dir if FLAGS.train_step > 1 else trained_dir
    epoch = model.restore_model_variables(
        checkpoint_dir=_checkpoint_dir, training_for_tf25=True, pop_key=False, training_step=FLAGS.train_step
    )
    model.summary()

    history_fine = train_model.fit(initial_epoch=epoch, bash=FLAGS.bash, save_best_only=FLAGS.save_best_only,)

    # timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    timestamp = datetime.now().strftime("%m-%d-%Y")
    history_path = os.path.join(
        checkpoint_dir, "{}-history-{}.pack".format(timestamp, const.TRAIN_TEMPLATE.format(FLAGS.train_step))
    )
    with open(history_path, "wb") as handle:
        dump(history_fine["history"], handle)

    print("\nTest accuracy: ")
    train_model.evaluate(datasets=test_ds)

    if FLAGS.save_model and model:
        model.save(checkpoint_dir)


def getParam(arg):
    if "=" in arg:
        vars = arg.split("=")
        return vars[1]
    raise ("Parameter format is invalid.")


if __name__ == "__main__":
    start(main)
