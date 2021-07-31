from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf

from absl import logging
from utils.summary import k_summary


class TrainClassifier:
    def __init__(
        self,
        model,
        batch_size,
        summary_path,
        epoches,
        data_handler_steps,
        test_ds,
        hparams,
        moving_average_decay,
        callbacks,
    ):
        self.model = model
        self.summary_path = summary_path
        self.epoches = epoches
        self.data_handler_steps = data_handler_steps
        self.test_ds = test_ds
        self._hparams = dict(hparams)
        self.moving_average_decay = moving_average_decay
        self.callbacks = callbacks
        self.train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
        self.regularization_loss_metric = tf.keras.metrics.Mean(name="regularization_loss")
        self.boundary_loss_metric = tf.keras.metrics.Mean(name="boundary_loss")
        self.total_loss_metric = tf.keras.metrics.Mean(name="total_loss")
        self.accuracy_metric = tf.keras.metrics.CategoricalAccuracy(name="accuracy")
        self.metrics = [
            self.train_loss_metric,
            self.regularization_loss_metric,
            self.boundary_loss_metric,
            self.total_loss_metric,
            self.accuracy_metric,
        ]
        self.batch_size = batch_size

        self.model.compile(self.metrics)

        if self.moving_average_decay:
            self.variable_averages = tf.train.ExponentialMovingAverage(self.moving_average_decay)
            self.model.variable_averages = self.variable_averages
            self.variable_averages.apply(model.variables)

        for callback in self.callbacks:
            callback.set_model(self.model.model)

    def get_average(self, var):
        if self.variable_averages:
            return self.variable_averages.average(var)

    @tf.function
    def train_step(self, inputs, labels):
        boundary_loss = 0.0
        with tf.GradientTape() as tape:
            predictions = self.model.process_step(inputs, training=True)
            if hasattr(self.model, "boundary_loss") and self.model.boundary_loss:
                boundary_loss = self.model.boundary_loss(inputs, training=True)
            train_loss = self.model.get_loss(labels, predictions)
            regularization_loss = tf.reduce_sum(self.model.get_regularization_loss())
            # if self.fine_grain_step < 2000:
            #     total_loss = train_loss
            # else:
            total_loss = regularization_loss + train_loss + boundary_loss

        if hasattr(self, "variable_averages") and self.variable_averages:
            self.variable_averages.apply(self.model.trainable_variables)
        gradients = tape.gradient(total_loss, self.model.trainable_variables)

        update_scale = tf.linalg.global_norm(gradients)
        param_scale = tf.linalg.global_norm(self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        ratio_of_weights_updates = update_scale / param_scale
        k_summary.scalar_update("scalar/param_scale", param_scale, self.fine_grain_step)
        k_summary.scalar_update("scalar/update_scale", update_scale, self.fine_grain_step)
        k_summary.scalar_update("scalar/ratio_of_weights_updates", ratio_of_weights_updates, self.fine_grain_step)

        self.train_loss_metric.update_state(train_loss)
        self.regularization_loss_metric.update_state(regularization_loss)
        self.boundary_loss_metric.update_state(boundary_loss)
        self.total_loss_metric.update_state(total_loss)
        self.accuracy_metric.update_state(labels, predictions)

        return {
            # "train_loss": self.train_loss_metric.result(),
            # "reg_loss": self.regularization_loss_metric.result(),
            # "b_loss": self.boundary_loss_metric.result(),
            "total_loss": self.total_loss_metric.result(),
            "accuracy": self.accuracy_metric.result(),
        }

    @tf.function
    def evaluate_step(self, inputs, labels):
        predictions = self.model.process_step(inputs, training=False)
        total_loss = self.model.get_loss(labels, predictions)
        self.total_loss_metric.update_state(total_loss)
        self.accuracy_metric.update_state(labels, predictions)
        return {"loss": self.total_loss_metric.result(), "accuracy": self.accuracy_metric.result()}

    def reset_metric(self):
        self.train_loss_metric.reset_states()
        self.regularization_loss_metric.reset_states()
        self.boundary_loss_metric.reset_states()
        self.total_loss_metric.reset_states()
        self.accuracy_metric.reset_states()

    def on_epoch_begin(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def fit(self, initial_epoch, **kwargs):
        target = self.data_handler_steps.size // self.batch_size
        self.fine_grain_step = tf.Variable(max(0, (initial_epoch - 1)) * target, dtype=tf.int64)
        is_run_from_bash = kwargs.pop("bash") if "bash" in kwargs else False
        # save_best_only = kwargs.pop("save_best_only") if "save_best_only" in kwargs else False
        finalize = False if not is_run_from_bash else True
        progbar = tf.keras.utils.Progbar(
            target,
            width=30,
            verbose=1,
            interval=0.05,
            stateful_metrics={"train_loss", "reg_loss", "b_loss", "total_loss", "accuracy"},
            unit_name="step",
        )

        # best_acc = 0
        first_graph_writing = True

        k_summary.re_init(self.summary_path, target)
        k_summary.hparams_pb(self._hparams)

        global_step = tf.compat.v1.train.get_or_create_global_step()
        for epoch in range(initial_epoch, self.epoches + 1):
            global_step.assign(epoch)
            print("\nEpoch: {}/{}".format(epoch, self.epoches))

            self.on_epoch_begin(epoch=epoch)
            k_summary.set_epoch(epoch=epoch)
            k_summary.hparams(self._hparams)

            self.reset_metric()
            seen = 0

            for inputs, labels in self.data_handler_steps:
                if inputs.shape.as_list()[0] == self.batch_size:
                    tfv = tf.version.VERSION.split(".")
                    if first_graph_writing and epoch == 1 and (tfv[0] == "2" and tfv[1] == "4"):
                        k_summary.trace_on(graph=True)
                    logs = self.train_step(inputs, labels)
                    logs = copy.copy(logs) if logs else {}
                    num_steps = logs.pop("num_steps", 1)
                    seen += num_steps
                    progbar.update(seen, list(logs.items()), finalize=finalize)

                if first_graph_writing and epoch == 1:
                    k_summary.graph(
                        fn_name="train_step", graph=self.train_step.get_concrete_function(inputs, labels).graph,
                    )
                    first_graph_writing = False

                accuracy = self.accuracy_metric.result().numpy()
                train_loss = self.train_loss_metric.result().numpy()
                k_summary.scalar_update("accuracy/fine_grain", accuracy, self.fine_grain_step)
                k_summary.scalar_update("loss/fine_grain_loss", train_loss, self.fine_grain_step)
                self.fine_grain_step.assign_add(1)
                k_summary.end_step()

            train_loss = self.train_loss_metric.result().numpy()
            regularization_loss = self.regularization_loss_metric.result().numpy()
            boundary_loss = self.boundary_loss_metric.result().numpy()
            total_loss = self.total_loss_metric.result().numpy()
            accuracy = self.accuracy_metric.result().numpy()

            # if epoch % 5 == 0:
            #     print("\n")
            #     self.reset_metric()
            #     t_acc, t_loss = self.evaluate(datasets=self.test_ds)

            #     overfitting = accuracy - t_acc
            #     self.model.overfitting.assign(overfitting)

            #     if overfitting > 0.1:
            #         self.model.fast_augment.assign(0)
            #     else:
            #         self.model.fast_augment.assign(1)

            #     k_summary.scalar_update("accuracy/validation", t_acc, epoch)
            #     k_summary.scalar_update("loss/validation/loss", t_loss, epoch)
            #     k_summary.scalar_update("accuracy/overfitting", overfitting, epoch)

            k_summary.scalar_update("accuracy/accuracy", accuracy, epoch)
            k_summary.scalar_update("loss/train_loss", train_loss, epoch)
            k_summary.scalar_update("loss/regularization_loss", regularization_loss, epoch)
            k_summary.scalar_update("loss/boundary_loss", boundary_loss, epoch)
            k_summary.scalar_update("loss/total_loss", total_loss, epoch)
            k_summary.scalar_update("scalar/learning_rate", self.model.optimizer.lr, epoch)

            k_summary.end_epoch()

            # if save_best_only and best_acc < t_acc:
            #     logging.info("\nSaving best weights...")
            #     best_acc = t_acc
            #     self.model.save_model_variables()
            # else:
            self.model.save_model_variables()

        k_summary.session_end_pb()

        print("\n")
        self.reset_metric()
        self.evaluate(datasets=self.test_ds)

    def evaluate(self, datasets, **kwargs):
        logs = None
        seen = 0
        target = datasets.size // self.batch_size
        is_run_from_bash = kwargs.pop("bash") if "bash" in kwargs else False
        finalize = False if not is_run_from_bash else True
        progbar = tf.keras.utils.Progbar(
            target, width=30, verbose=1, interval=0.05, stateful_metrics={"loss", "accuracy"}, unit_name="step"
        )
        for inputs, labels in datasets:
            if inputs.shape.as_list()[0] == self.batch_size:
                logs = self.evaluate_step(inputs, labels)
                num_steps = logs.pop("num_steps", 1)
                seen += num_steps
                progbar.update(seen, list(logs.items()), finalize=finalize)
        logs = copy.copy(logs) if logs else {}
        print("\nloss: {:.3f}, accuracy: {:.3f}".format(logs["loss"], logs["accuracy"]))

        return logs["accuracy"], logs["loss"]
