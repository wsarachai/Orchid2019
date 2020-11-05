from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf

from data import data_utils, orchids52_dataset
from data.data_utils import dataset_mapping
from lib_utils import start, latest_checkpoint
from nets import nets_utils
from nets.mobilenet_v2 import create_mobilenet_v2
from nets.mobilenet_v2_140_orchids52 import create_predict_module
from nets.nets_utils import TRAIN_V2_STEP2, TRAIN_TEMPLATE, TRAIN_STEP4

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

flags.DEFINE_integer('total_epochs', 100,
                     'Total epochs')


class TrainClassifier:

    def __init__(self, model, learning_rate, batch_size):
        self.model = model
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.loss_metric = tf.keras.metrics.Mean(name='train_loss')
        self.accuracy_metric = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
        self.batch_size = batch_size

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
                else:
                    logging.error('\n{epoch}: Error batch size {b1} != {b2}.'.format(
                        epoch=epoch,
                        b1=batch_size,
                        b2=inputs.shape.as_list()[0]
                    ))

            self.reset_metric()

            for inputs, labels in validate_ds:
                if inputs.shape.as_list()[0] == self.batch_size:
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


def main(unused_argv):
    logging.debug(unused_argv)
    load_dataset = dataset_mapping[data_utils.ORCHIDS52_V1_TFRECORD]
    create_model = nets_utils.nets_mapping[nets_utils.MOBILENET_V2_140_ORCHIDS52]

    for train_step in range(1, 5):
        if train_step == 1:
            batch_size = FLAGS.batch_size
        else:
            batch_size = FLAGS.batch_size // 4

        train_ds = load_dataset(split="train", batch_size=batch_size)
        validate_ds = load_dataset(split="validate", batch_size=batch_size)
        test_ds = load_dataset(split="test", batch_size=batch_size)

        training_step = TRAIN_TEMPLATE.format(step=train_step)
        model = create_model(num_classes=orchids52_dataset.NUM_OF_CLASSES,
                             is_training=True,
                             batch_size=batch_size,
                             step=training_step)

        latest, epoch = latest_checkpoint(FLAGS.checkpoint_path, training_step)
        if latest:
            model.resume_model_weights(latest)
        else:
            model.load_model_weights(FLAGS.checkpoint_path, epoch)
            epoch = 0

        model.summary()

        if FLAGS.exp_decay:
            base_learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.001,
                decay_steps=10,
                decay_rate=0.96
            )
        else:
            base_learning_rate = 0.001
            if training_step in [TRAIN_STEP4, TRAIN_V2_STEP2]:
                base_learning_rate = 0.00001

        train_model = TrainClassifier(model=model, learning_rate=base_learning_rate, batch_size=batch_size)

        train_model.fit(initial_epoch=epoch,
                        epoches=FLAGS.total_epochs,
                        train_ds=train_ds,
                        validate_ds=validate_ds,
                        batch_size=batch_size,
                        checkpoint_path=FLAGS.checkpoint_path)

        print('Test accuracy: ')
        train_model.evaluate(datasets=test_ds, batch_size=batch_size)


if __name__ == '__main__':
    tf.config.experimental_run_functions_eagerly(True)
    start(main)