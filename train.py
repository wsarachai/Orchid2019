from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os

import tensorflow as tf
from tensorflow.keras import layers, applications
from tensorflow.keras.utils import Progbar
from tensorflow.python.keras import Sequential, Input

from data import data_utils, orchids52_dataset
from data.data_utils import dataset_mapping
from lib_utils import start, latest_checkpoint, get_checkpoint_file
from nets import nets_utils
from nets.mobilenet_v2 import create_mobilenet_v2
from nets.nets_utils import TRAIN_STEP1, TRAIN_STEP2, TRAIN_STEP3, TRAIN_V2_STEP2, TRAIN_V2_STEP1, TRAIN_TEMPLATE

flags = tf.compat.v1.flags
logging = tf.compat.v1.logging
FLAGS = flags.FLAGS

flags.DEFINE_string('tf_record_dir', '/Volumes/Data/_dataset/_orchids_dataset/orchids52_data/v1/tf-records',
                    'TF record data directory')

flags.DEFINE_string('checkpoint_path', '/Volumes/Data/tmp/orchids-models/orchid2019',
                    'The checkpoint path')

flags.DEFINE_boolean('exp_decay', False,
                     'Exponential decay learning rate')

flags.DEFINE_integer('batch_size', 1,
                     'Batch size')

flags.DEFINE_integer('total_epochs', 11,
                     'Total epochs')


class TrainClassifier:

    def __init__(self, model):
        self.model = model
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.loss_metric = tf.keras.metrics.Mean(name='train_loss')
        self.accuracy_metric = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    @tf.function
    def train_step(self, inputs, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            total_loss = self.loss_fn(labels, predictions)

        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.loss_metric.update_state(total_loss)
        self.accuracy_metric.update_state(labels, predictions)

        return {
            'loss': self.loss_metric.result(),
            'accuracy': self.accuracy_metric.result()
        }

    @tf.function
    def evaluate_step(self, inputs, labels):
        predictions = self.model(inputs, training=True)
        total_loss = self.loss_fn(labels, predictions)
        self.loss_metric.update_state(total_loss)
        self.accuracy_metric.update_state(labels, predictions)
        return {
            'loss': self.loss_metric.result(),
            'accuracy': self.accuracy_metric.result()
        }

    def reset_metric(self):
        self.loss_metric.reset_states()
        self.accuracy_metric.reset_states()

    def fit(self, initial_epoch, epoches, train_ds, validate_ds, batch_size, checkpoint_path):
        logs = None
        target = train_ds.size // batch_size
        progbar = tf.keras.utils.Progbar(
            target, width=30, verbose=1, interval=0.05, stateful_metrics={'loss', 'accuracy'},
            unit_name='step'
        )
        val_accuracy = 0.0
        val_loss = 1.0
        for epoch in range(initial_epoch, epoches):
            print('\nEpoch: ', epoch)

            self.reset_metric()
            seen = 0

            for inputs, labels in train_ds:
                if inputs.shape.as_list()[0] == batch_size:
                    logs = self.train_step(inputs, labels)
                    logs = copy.copy(logs) if logs else {}
                    num_steps = logs.pop('num_steps', 1)
                    seen += num_steps
                    progbar.update(seen, list(logs.items()), finalize=False)

            self.reset_metric()

            for inputs, labels in validate_ds:
                if inputs.shape.as_list()[0] == FLAGS.batch_size:
                    logs = self.evaluate_step(inputs, labels)

            logs = copy.copy(logs) if logs else {}
            print(', val_loss: {:.3f}, val_accuracy: {:.3f}'.format(logs['loss'], logs['accuracy']))

            if val_accuracy < logs['accuracy'].numpy() or val_loss > logs['loss'].numpy():
                self.model.save_model_weights(checkpoint_path, epoch)

    def evaluate(self, datasets, batch_size):
        logs = None
        for inputs, labels in datasets:
            if inputs.shape.as_list()[0] == batch_size:
                logs = self.evaluate_step(inputs, labels)
        logs = copy.copy(logs) if logs else {}
        print('loss: {:.3f}, accuracy: {:.3f}'.format(logs['loss'], logs['accuracy']))


def load_weight(checkpoint_path, training_step, model):
    if training_step == TRAIN_STEP1:
        model.config_layers(TRAIN_STEP1)
    elif FLAGS.training_step == TRAIN_STEP2:
        _, step = latest_checkpoint(TRAIN_STEP1, checkpoint_path)
        model.load_model_weights(checkpoint_path, step, by_name=True, skip_mismatch=True)
    elif FLAGS.training_step == TRAIN_STEP3:
        latest, _ = latest_checkpoint(TRAIN_STEP2)
        model.load_weights(latest, by_name=True, skip_mismatch=True)
    elif FLAGS.training_step == TRAIN_V2_STEP2:
        latest, _ = latest_checkpoint(TRAIN_V2_STEP1)
        model.load_weights(latest, by_name=True, skip_mismatch=True)


def main(unused_argv):
    logging.debug(unused_argv)
    load_dataset = dataset_mapping[data_utils.ORCHIDS52_V1_TFRECORD]
    train_ds = load_dataset(
        split="train",
        repeat=True,
        batch_size=FLAGS.batch_size)
    validate_ds = load_dataset(
        split="validate",
        repeat=True,
        batch_size=FLAGS.batch_size)
    test_ds = load_dataset(
        split="test",
        repeat=True,
        batch_size=FLAGS.batch_size)

    create_model = nets_utils.nets_mapping[nets_utils.MOBILENET_V2_140_ORCHIDS52]

    for train_step in range(1, 3):
        training_step = TRAIN_TEMPLATE.format(step=train_step)
        model = create_model(num_classes=orchids52_dataset.NUM_OF_CLASSES,
                             is_training=True,
                             batch_size=FLAGS.batch_size,
                             step=training_step)

        latest, epoch = latest_checkpoint(training_step)
        if latest:
            model.resume_weights(latest)
        else:
            epoch = 0
            load_weight(FLAGS.checkpoint_path, training_step, model)

        model.summary()

        train_model = TrainClassifier(model=model)

        train_model.fit(initial_epoch=epoch,
                        epoches=FLAGS.total_epochs,
                        train_ds=train_ds,
                        validate_ds=validate_ds,
                        batch_size=FLAGS.batch_size,
                        checkpoint_path=FLAGS.checkpoint_path)

        print('Test accuracy: ')
        train_model.evaluate(datasets=test_ds, batch_size=FLAGS.batch_size)


if __name__ == '__main__':
    tf.config.experimental_run_functions_eagerly(True)
    start(main)
