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
    if not tf.io.gfile.exists(training_dir):
        os.makedirs(training_dir)

    train_ds = load_dataset(
        flags=FLAGS, workspace_path=workspace_path, split="train", preprocessing=True, one_hot=True
    )
    test_ds = load_dataset(flags=FLAGS, workspace_path=workspace_path, split="test", preprocessing=True, one_hot=True)

    train_size = train_ds.size
    test_size = test_ds.size
    num_of_classes = train_ds.num_of_classes

    train_ds = train_ds.map(lambda img, lbl: (preprocess_image(img, width=224, height=224, is_training=True), lbl))
    test_ds = test_ds.map(lambda img, lbl: (preprocess_image(img, width=224, height=224, is_training=False), lbl))

    train_ds = train_ds.batch(batch_size=FLAGS.batch_size)
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.batch(batch_size=FLAGS.batch_size)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    train_ds.size = train_size
    test_ds.size = test_size
    train_ds.num_of_classes = num_of_classes

    learning_rate_schedule = config_learning_rate(
        learning_rate=FLAGS.learning_rate,
        num_epochs_per_decay=2,
        batch_size=FLAGS.batch_size,
        num_samples_per_epoch=train_ds.size,
        decay=FLAGS.learning_rate_decay,
    )
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
        dropout=FLAGS.dropout,
    )

    def scheduler(epochs, _):
        # tf.keras.optimizers.schedules.InverseTimeDecay
        learning_rate = 0.000004
        _epochs = epochs
        if _epochs > 3:
            if FLAGS.fine_tune:
                _epochs = _epochs - epoch
            learning_rate = learning_rate_schedule(_epochs)
        return learning_rate

    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    log_dir = os.path.join(training_dir, "logs", const.TRAIN_TEMPLATE.format(FLAGS.train_step))

    train_model = TrainClassifier(
        model=model,
        batch_size=FLAGS.batch_size,
        summary_path=log_dir,
        epoches=FLAGS.total_epochs,
        data_handler_steps=train_ds,
        test_ds=test_ds,
        # moving_average_decay=FLAGS.moving_average_decay,
        moving_average_decay=None,
        callbacks=[callback],
        hparams={
            "model": FLAGS.model,
            "dataset": FLAGS.dataset,
            "training_state": const.TRAIN_TEMPLATE.format(FLAGS.train_step),
            "learning_rate": FLAGS.learning_rate,
            "optimizer": FLAGS.optimizer,
            "weight_decay": FLAGS.learning_rate_decay,
            "batch_size": FLAGS.batch_size,
            "dropout": FLAGS.dropout,
            "epoches": FLAGS.total_epochs,
        },
    )

    model.config_layers(fine_tune=FLAGS.fine_tune, fine_tune_at=FLAGS.fine_tune_at)
    for var in model.trainable_variables:
        logging.info("trainable variable: %s", var.name)

    # average_vars = []
    # for v in model.variables:
    #     if "moving" not in v.name:
    #         average_vars.append(train_model.get_average(v))

    model.config_checkpoint(training_dir)
    _checkpoint_dir = training_dir if FLAGS.train_step > 1 else trained_weights_dir
    epoch = model.restore_model_variables(
        checkpoint_dir=_checkpoint_dir,
        training_for_tf25=True,
        pop_key=False,
        training_step=FLAGS.train_step,
        # average_vars=average_vars,
    )

    model.summary()

    train_model.fit(
        initial_epoch=epoch,
        bash=FLAGS.bash,
        save_best_only=FLAGS.save_best_only,
    )

    # if FLAGS.save_model and model:
    #     model.save(training_dir)


def getParam(arg):
    if "=" in arg:
        vars = arg.split("=")
        return vars[1]
    raise ("Parameter format is invalid.")


if __name__ == "__main__":
    start(main)
