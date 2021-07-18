from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from absl import logging
from nets.mapping import nets_mapping
from data.data_utils import load_dataset
from utils.lib_utils import config_learning_rate
from utils.lib_utils import config_optimizer
from utils.lib_utils import config_loss
from utils.start_app import FLAGS, start
from utils.training_utils import TrainClassifier
from utils import const


def main(unused_argv):
    logging.debug(unused_argv)

    # tf.config.run_functions_eagerly(True)

    create_model_graph = False

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

    model.config_checkpoint(training_dir)
    _checkpoint_dir = training_dir if FLAGS.train_step > 1 else trained_weights_dir
    epoch = model.restore_model_variables(
        checkpoint_dir=_checkpoint_dir, training_for_tf25=True, pop_key=False, training_step=FLAGS.train_step
    )
    model.summary()

    def scheduler(epochs, _):
        return learning_rate_schedule(epochs)

    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    if create_model_graph:
        log_dir = os.path.join(training_dir, "graph", const.TRAIN_TEMPLATE.format(FLAGS.train_step))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        model.compile(metrics=["accuracy"])

        model.fit(
            train_ds, initial_epoch=0, epochs=1, callbacks=[callback, tensorboard_callback],
        )

    log_dir = os.path.join(training_dir, "logs", const.TRAIN_TEMPLATE.format(FLAGS.train_step))

    train_model = TrainClassifier(
        model=model,
        batch_size=FLAGS.batch_size,
        summary_path=log_dir,
        epoches=FLAGS.total_epochs,
        data_handler_steps=train_ds,
        callbacks=[callback],
        hparams={
            "model": FLAGS.model,
            "dataset": FLAGS.dataset,
            "training_state": const.TRAIN_TEMPLATE.format(FLAGS.train_step),
            "learning_rate": FLAGS.learning_rate,
            "optimizer": FLAGS.optimizer,
            "weight_decay": FLAGS.learning_rate_decay,
            "batch_size": FLAGS.batch_size,
            "dropout": 0.2,
            "epoches": FLAGS.total_epochs,
        },
    )

    train_model.fit(
        initial_epoch=epoch, bash=FLAGS.bash, save_best_only=FLAGS.save_best_only,
    )

    print("\nTest accuracy: ")
    train_model.evaluate(datasets=test_ds)

    if FLAGS.save_model and model:
        model.save(training_dir)


def getParam(arg):
    if "=" in arg:
        vars = arg.split("=")
        return vars[1]
    raise ("Parameter format is invalid.")


if __name__ == "__main__":
    start(main)
