from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import tensorflow as tf
from nets import constants

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

logging = tf.compat.v1.logging


def get_checkpoint_file(checkpoint_dir, name):
    if isinstance(name, int):
        return os.path.join(checkpoint_dir, 'cp-{name:04d}'.format(name=name))
    else:
        return os.path.join(checkpoint_dir, 'cp-{name}'.format(name=name))


def get_step_number(checkpoint_dir):
    idx = checkpoint_dir.index('.')
    step = int(checkpoint_dir[:idx][-4:])
    return step


def apply_with_random_selector(x, func, num_cases):
    sel = tf.random.uniform([], maxval=num_cases, dtype=tf.int32)
    return tf.raw_ops.Merge(inputs=[func(tf.raw_ops.Switch(data=x, pred=tf.equal(sel, case))[1], case)
                                    for case in range(num_cases)])[0]


def start(start_fn):
    logging.set_verbosity(logging.INFO)
    logging.info("tf.version %s" % tf.version.VERSION)
    tf.compat.v1.app.run(start_fn)


def config_learning_rate(learning_rate=0.001,
                         exp_decay=False,
                         **kwargs):
    training_step = kwargs.pop('training_step') if 'training_step' in kwargs else ''
    if exp_decay:
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=10,
            decay_rate=0.96
        )
    else:
        if training_step in [constants.TRAIN_STEP4, constants.TRAIN_V2_STEP2]:
            learning_rate = 0.00001
    return learning_rate


def config_optimizer(optimizer, learning_rate, **kwargs):
    if optimizer == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        return tf.keras.optimizers.SGD(learning_rate=learning_rate)


def config_loss(from_logits):
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)
    return loss_fn


class TrainClassifier:

    def __init__(self, model, batch_size):
        self.model = model
        self.train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
        self.regularization_loss_metric = tf.keras.metrics.Mean(name='regularization_loss')
        self.boundary_loss_metric = tf.keras.metrics.Mean(name='boundary_loss')
        self.total_loss_metric = tf.keras.metrics.Mean(name='total_loss')
        self.accuracy_metric = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
        self.batch_size = batch_size

        self.model.compile()

    def train_step(self, inputs, labels):
        boundary_loss = 0.
        with tf.GradientTape() as tape:
            predictions = self.model.process_step(inputs, training=True)
            if hasattr(self.model, 'boundary_loss') and self.model.boundary_loss:
                boundary_loss = self.model.boundary_loss(inputs, training=True)
            train_loss = self.model.get_loss(labels, predictions)
            regularization_loss = tf.reduce_sum(self.model.get_regularization_loss())
            total_loss = regularization_loss + train_loss + boundary_loss

        gradients = tape.gradient(total_loss, self.model.get_trainable_variables())
        self.model.optimizer.apply_gradients(zip(gradients, self.model.get_trainable_variables()))

        self.train_loss_metric.update_state(train_loss)
        self.regularization_loss_metric.update_state(regularization_loss)
        self.boundary_loss_metric.update_state(boundary_loss)
        self.total_loss_metric.update_state(total_loss)
        self.accuracy_metric.update_state(labels, predictions)

        return {
            'train_loss': train_loss,
            'reg_loss': regularization_loss,
            'b_loss': boundary_loss,
            'total_loss': total_loss,
            'accuracy': self.accuracy_metric.result()
        }

    def evaluate_step(self, inputs, labels):
        predictions = self.model.process_step(inputs, training=False)
        total_loss = self.model.get_loss(labels, predictions)
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
            **kwargs):
        history = {
            'train_loss': [],
            'reg_loss': [],
            'b_loss': [],
            'total_loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        target = train_ds.size // self.batch_size
        is_run_from_bash = kwargs.pop('bash') if 'bash' in kwargs else False
        save_best_only = kwargs.pop('save_best_only') if 'save_best_only' in kwargs else False
        finalize = False if not is_run_from_bash else True
        progbar = tf.keras.utils.Progbar(
            target, width=30, verbose=1, interval=0.05,
            stateful_metrics={'train_loss', 'reg_loss', 'b_loss', 'total_loss', 'accuracy'},
            unit_name='step'
        )
        val_accuracy = 0.0
        val_loss = 1.0
        for epoch in range(initial_epoch, epoches + 1):
            print('\nEpoch: {}/{}'.format(epoch, epoches))

            self.reset_metric()
            seen = 0

            for inputs, labels in train_ds:
                if inputs.shape.as_list()[0] == self.batch_size:
                    logs = self.train_step(inputs, labels)
                    logs = copy.copy(logs) if logs else {}
                    num_steps = logs.pop('num_steps', 1)
                    seen += num_steps
                    progbar.update(seen, list(logs.items()), finalize=finalize)
                # else:
                #     logging.error('\n{epoch}: Error batch size {b1} != {b2}.'.format(
                #         epoch=epoch,
                #         b1=batch_size,
                #         b2=inputs.shape.as_list()[0]
                #     ))

            history['train_loss'].append(self.train_loss_metric.result().numpy())
            history['reg_loss'].append(self.regularization_loss_metric.result().numpy())
            history['b_loss'].append(self.boundary_loss_metric.result().numpy())
            history['total_loss'].append(self.total_loss_metric.result().numpy())
            history['accuracy'].append(self.accuracy_metric.result().numpy())

            self.reset_metric()

            logs = None
            for inputs, labels in validate_ds:
                if inputs.shape.as_list()[0] == self.batch_size:
                    logs = self.evaluate_step(inputs, labels)

            logs = copy.copy(logs) if logs else {}
            history['val_loss'].append(logs['loss'].numpy())
            history['val_accuracy'].append(logs['accuracy'].numpy())
            print('\nValidation: val_loss: {:.3f}, val_accuracy: {:.3f}\n'.format(logs['loss'], logs['accuracy']))

            if save_best_only and val_accuracy < logs['accuracy'].numpy() or val_loss > logs['loss'].numpy():
                val_accuracy = logs['accuracy'].numpy()
                val_loss = logs['loss'].numpy()
                self.model.save_model_variables()
            else:
                self.model.save_model_variables()
        return {
            'history': history
        }

    def evaluate(self, datasets, **kwargs):
        logs = None
        seen = 0
        target = datasets.size // self.batch_size
        is_run_from_bash = kwargs.pop('bash') if 'bash' in kwargs else False
        finalize = False if not is_run_from_bash else True
        progbar = tf.keras.utils.Progbar(
            target, width=30, verbose=1, interval=0.05,
            stateful_metrics={'loss', 'accuracy'},
            unit_name='step'
        )
        for inputs, labels in datasets:
            if inputs.shape.as_list()[0] == self.batch_size:
                logs = self.evaluate_step(inputs, labels)
                num_steps = logs.pop('num_steps', 1)
                seen += num_steps
                progbar.update(seen, list(logs.items()), finalize=finalize)
        logs = copy.copy(logs) if logs else {}
        print('loss: {:.3f}, accuracy: {:.3f}\n'.format(logs['loss'], logs['accuracy']))
