from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import tensorflow as tf
from data import data_utils, orchids52_dataset
from data.data_utils import dataset_mapping
from lib_utils import start, latest_checkpoint
from nets import nets_utils
from nets.nets_utils import TRAIN_V2_STEP2, TRAIN_TEMPLATE, TRAIN_STEP4

flags = tf.compat.v1.flags
logging = tf.compat.v1.logging
FLAGS = flags.FLAGS

flags.DEFINE_boolean('exp_decay', False,
                     'Exponential decay learning rate')

flags.DEFINE_integer('batch_size', 32,
                     'Batch size')

flags.DEFINE_integer('total_epochs', 100,
                     'Total epochs')

flags.DEFINE_integer('start_state', 1,
                     'Start state')

flags.DEFINE_integer('end_state', 5,
                     'End state')

flags.DEFINE_float('learning_rate', 0.0001,
                   'Learning Rate')

flags.DEFINE_string('aug_method', 'fast',
                    'Augmentation Method')


class TrainClassifier:

    def __init__(self, model, learning_rate, batch_size):
        self.model = model
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                          reduction=tf.keras.losses.Reduction.NONE)
        self.train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
        self.regularization_loss_metric = tf.keras.metrics.Mean(name='regularization_loss')
        self.boundary_loss_metric = tf.keras.metrics.Mean(name='boundary_loss')
        self.total_loss_metric = tf.keras.metrics.Mean(name='total_loss')
        self.accuracy_metric = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
        self.batch_size = batch_size

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss_fn,
                           metrics=[
                               self.train_loss_metric,
                               self.regularization_loss_metric,
                               self.boundary_loss_metric,
                               self.total_loss_metric,
                               self.accuracy_metric
                           ])

    @tf.function
    def train_step(self, inputs, labels):
        boundary_loss = 0.
        train_loss = 0.
        regularization_loss = 0.
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            if self.model.boundary_loss:
                boundary_loss = self.model.boundary_loss(inputs, training=True)
            train_loss = self.loss_fn(labels, predictions)
            train_loss = tf.reduce_sum(train_loss) * (1. / self.batch_size)
            regularization_loss = tf.reduce_sum(self.model.losses)
            total_loss = regularization_loss + train_loss + boundary_loss

        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss_metric.update_state(train_loss)
        self.regularization_loss_metric.update_state(regularization_loss)
        self.boundary_loss_metric.update_state(boundary_loss)
        self.total_loss_metric.update_state(total_loss)
        self.accuracy_metric.update_state(labels, predictions)

        return {
            'train_loss': self.train_loss_metric.result(),
            'regularization_loss': self.regularization_loss_metric.result(),
            'boundary_loss': self.boundary_loss_metric.result(),
            'total_loss': self.total_loss_metric.result(),
            'accuracy': self.accuracy_metric.result()
        }

    @tf.function
    def evaluate_step(self, inputs, labels):
        predictions = self.model(inputs, training=False)
        total_loss = self.loss_fn(labels, predictions)
        self.total_loss_metric.update_state(total_loss)
        self.accuracy_metric.update_state(labels, predictions)
        return {
            'loss': self.total_loss_metric.result(),
            'accuracy': self.accuracy_metric.result()
        }

    def reset_metric(self):
        self.train_loss_metric.reset_states()
        self.regularization_loss_metric.reset_states()
        self.boundary_loss_metric.reset_states()
        self.total_loss_metric.reset_states()
        self.accuracy_metric.reset_states()

    def fit(self,
            initial_epoch,
            epoches,
            train_ds,
            validate_ds,
            test_ds,
            batch_size,
            checkpoint_path):
        logs = None
        target = train_ds.size // batch_size
        progbar = tf.keras.utils.Progbar(
            target, width=30, verbose=1, interval=0.05,
            stateful_metrics={'train_loss', 'regularization_loss', 'boundary_loss', 'total_loss', 'accuracy'},
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
                # else:
                #     logging.error('\n{epoch}: Error batch size {b1} != {b2}.'.format(
                #         epoch=epoch,
                #         b1=batch_size,
                #         b2=inputs.shape.as_list()[0]
                #     ))

            self.reset_metric()

            for inputs, labels in validate_ds:
                if inputs.shape.as_list()[0] == self.batch_size:
                    logs = self.evaluate_step(inputs, labels)
            logs = copy.copy(logs) if logs else {}
            print(', val_loss: {:.3f}, val_accuracy: {:.3f}'.format(logs['loss'], logs['accuracy']))

            if epoch % 20 == 0:
                for inputs, labels in test_ds:
                    if inputs.shape.as_list()[0] == self.batch_size:
                        logs = self.evaluate_step(inputs, labels)
                logs = copy.copy(logs) if logs else {}
                print('\ntest_loss: {:.3f}, test_accuracy: {:.3f}'.format(logs['loss'], logs['accuracy']))

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
    workspace_path = os.environ['WORKSPACE'] if 'WORKSPACE' in os.environ else '/Volumes/Data/tmp'
    data_path = os.environ['DATA_DIR'] if 'DATA_DIR' in os.environ else '/Volumes/Data/_dataset/_orchids_dataset'
    data_dir = os.path.join(data_path, 'orchids52_data')
    checkpoint_path = os.path.join(workspace_path, 'orchids-models', 'orchids2019')
    load_dataset = dataset_mapping[data_utils.ORCHIDS52_V1_TFRECORD]
    create_model = nets_utils.nets_mapping[nets_utils.MOBILENET_V2_140_ORCHIDS52]

    if not tf.io.gfile.exists(checkpoint_path):
        tf.io.gfile.mkdir(checkpoint_path)

    for train_step in range(FLAGS.start_state, FLAGS.end_state):
        if train_step == 1:
            batch_size = FLAGS.batch_size
        else:
            batch_size = FLAGS.batch_size // 4

        train_ds = load_dataset(split="train",
                                batch_size=batch_size,
                                root_path=data_dir,
                                aug_method=FLAGS.aug_method)
        validate_ds = load_dataset(split="validate", batch_size=batch_size, root_path=data_dir)
        test_ds = load_dataset(split="test", batch_size=batch_size, root_path=data_dir)

        training_step = TRAIN_TEMPLATE.format(step=train_step)
        model = create_model(num_classes=orchids52_dataset.NUM_OF_CLASSES,
                             is_training=True,
                             batch_size=batch_size,
                             step=training_step)

        if FLAGS.exp_decay:
            base_learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=FLAGS.learning_rate,
                decay_steps=10,
                decay_rate=0.96
            )
        else:
            base_learning_rate = FLAGS.learning_rate
            if training_step in [TRAIN_STEP4, TRAIN_V2_STEP2]:
                base_learning_rate = 0.00001

        train_model = TrainClassifier(model=model, learning_rate=base_learning_rate, batch_size=batch_size)

        latest, epoch = latest_checkpoint(checkpoint_path, training_step)
        if latest:
            model.resume_model_weights(latest)
        else:
            model.load_model_weights(checkpoint_path, epoch)
            epoch = 0

        model.summary()

        train_model.fit(initial_epoch=epoch,
                        epoches=FLAGS.total_epochs,
                        train_ds=train_ds,
                        validate_ds=validate_ds,
                        test_ds=test_ds,
                        batch_size=batch_size,
                        checkpoint_path=checkpoint_path)

        print('Test accuracy: ')
        train_model.evaluate(datasets=test_ds, batch_size=batch_size)

        model_path = os.path.join(checkpoint_path, str(train_step))
        if not tf.io.gfile.exists(model_path):
            tf.io.gfile.mkdir(model_path)
        model.save(model_path)


if __name__ == '__main__':
    tf.config.experimental_run_functions_eagerly(True)
    start(main)
