from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from datetime import datetime
from pickle import dump
from absl import logging
from data import orchids52_dataset
from data.data_utils import dataset_mapping
from nets.mapping import nets_mapping
from utils.lib_utils import FLAGS
from utils.lib_utils import config_learning_rate
from utils.lib_utils import config_optimizer
from utils.lib_utils import config_loss
from utils.lib_utils import start
from utils.training_utils import TrainClassifier
from utils.const import TRAIN_TEMPLATE


def main(unused_argv):
    logging.debug(unused_argv)
    workspace_path = os.environ["WORKSPACE"] if "WORKSPACE" in os.environ else "/Volumes/Data/tmp"
    data_path = os.environ["DATA_DIR"] if "DATA_DIR" in os.environ else "/Users/watcharinsarachai/Documents/_datasets/"
    data_dir = os.path.join(data_path, "orchids52_data")
    load_dataset = dataset_mapping[FLAGS.dataset]
    create_model = nets_mapping[FLAGS.model]
    checkpoint_dir = os.path.join(workspace_path, "orchids-models", "orchids2019", FLAGS.model)

    if not tf.io.gfile.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

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

        train_ds = load_dataset(split="train", batch_size=batch_size, root_path=data_dir)
        validate_ds = load_dataset(split="validate", batch_size=batch_size, root_path=data_dir)
        test_ds = load_dataset(split="test", batch_size=batch_size, root_path=data_dir)

        training_step = TRAIN_TEMPLATE.format(step=train_step)

        learning_rate = config_learning_rate(
            learning_rate=FLAGS.learning_rate, exp_decay=FLAGS.exp_decay, training_step=training_step
        )
        optimizer = config_optimizer(FLAGS.optimizer, learning_rate=learning_rate, training_step=training_step)
        loss_fn = config_loss()

        model = create_model(
            num_classes=orchids52_dataset.NUM_OF_CLASSES,
            optimizer=optimizer,
            loss_fn=loss_fn,
            training=True,
            batch_size=batch_size,
            step=training_step,
        )

        train_model = TrainClassifier(model=model, batch_size=batch_size)

        model.config_checkpoint(checkpoint_dir)
        epoch = model.restore_model_variables(checkpoint_dir=FLAGS.trained_dir)
        model.summary()

        history_fine = train_model.fit(
            initial_epoch=epoch,
            epoches=total_epochs[idx],
            train_ds=train_ds,
            validate_ds=validate_ds,
            bash=FLAGS.bash,
            save_best_only=FLAGS.save_best_only,
        )

        timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        history_path = os.path.join(checkpoint_dir, "{}-history-{}.pack".format(timestamp, training_step))
        with open(history_path, "wb") as handle:
            dump(history_fine["history"], handle)

        print("Test accuracy: ")
        train_model.evaluate(datasets=test_ds)

    if FLAGS.save_model and model:
        model.save(checkpoint_dir)


if __name__ == "__main__":
    start(main)
