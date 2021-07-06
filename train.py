from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from absl import logging
from nets.mapping import nets_mapping
from nets.mapping import preprocessing_mapping
from data.data_utils import load_dataset
from utils.lib_utils import FLAGS
from utils.lib_utils import start
from utils.lib_utils import FLAGS
from utils.lib_utils import config_learning_rate
from utils.lib_utils import config_optimizer
from utils.lib_utils import config_loss
from utils.lib_utils import TrainClassifier
from utils.lib_utils import start
from utils import const


def main(unused_argv):
    logging.debug(unused_argv)

    split = "train"
    workspace_path = os.environ["WORKSPACE"] if "WORKSPACE" in os.environ else "/Users/watcharinsarachai/Documents/"
    image_dir = os.path.join(
        workspace_path, "_datasets", FLAGS.dataset, FLAGS.dataset_format, FLAGS.dataset_version, split
    )

    train_ds = load_dataset(flags=FLAGS, workspace_path=workspace_path, split=split, preprocessing=True, one_hot=True)

    create_model = nets_mapping[FLAGS.model]

    checkpoint_path = os.path.join(workspace_path, "_trained_models", "orchids2019", FLAGS.model)
    if not tf.io.gfile.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    model = None
    total_epochs = [int(e) for e in FLAGS.total_epochs.split(",")]
    num_state = FLAGS.end_state - FLAGS.start_state
    assert num_state == len(total_epochs)
    for idx, train_step in enumerate(range(FLAGS.start_state, FLAGS.end_state)):
        if train_step == 1:
            batch_size = FLAGS.batch_size
        elif train_step == 4:
            batch_size = FLAGS.batch_size // 8
        else:
            batch_size = FLAGS.batch_size // 4

        training_step = const.TRAIN_TEMPLATE.format(train_step)

        learning_rate = config_learning_rate(
            learning_rate=FLAGS.learning_rate, exp_decay=FLAGS.exp_decay, training_step=training_step
        )
        optimizer = config_optimizer(FLAGS.optimizer, learning_rate=learning_rate, training_step=training_step)
        loss_fn = config_loss()

        model = create_model(
            num_classes=train_ds.num_of_classes,
            optimizer=optimizer,
            loss_fn=loss_fn,
            training=True,
            step=training_step,
            activation="softmax",
        )

        train_model = TrainClassifier(model=model, batch_size=batch_size)

        model.config_checkpoint(checkpoint_path)
        epoch = model.restore_model_variables(checkpoint_path=FLAGS.trained_path)
        model.summary()

        history_fine = train_model.fit(
            initial_epoch=epoch,
            epoches=total_epochs[idx],
            train_ds=train_ds,
            bash=FLAGS.bash,
            save_best_only=FLAGS.save_best_only,
        )

        model.config_checkpoint(checkpoint_path)
        epoch = model.restore_model_variables(checkpoint_path=FLAGS.trained_path)
        model.summary()

    if FLAGS.save_model and model:
        model.save(checkpoint_path)


if __name__ == "__main__":
    # tf.config.run_functions_eagerly(True)
    start(main)
