from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf


class TrainClassifier:
    def __init__(self, model, batch_size, summary_path):
        self.model = model
        self.summary_path = summary_path
        self.train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
        self.regularization_loss_metric = tf.keras.metrics.Mean(name="regularization_loss")
        self.boundary_loss_metric = tf.keras.metrics.Mean(name="boundary_loss")
        self.total_loss_metric = tf.keras.metrics.Mean(name="total_loss")
        self.accuracy_metric = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")
        self.metrics = [
            self.train_loss_metric,
            self.regularization_loss_metric,
            self.boundary_loss_metric,
            self.total_loss_metric,
            self.accuracy_metric,
        ]
        self.batch_size = batch_size

        self.model.compile(self.metrics)

    def train_step(self, inputs, labels):
        boundary_loss = 0.0
        with tf.GradientTape() as tape:
            predictions = self.model.process_step(inputs, training=True)
            if hasattr(self.model, "boundary_loss") and self.model.boundary_loss:
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
            "train_loss": train_loss,
            "reg_loss": regularization_loss,
            "b_loss": boundary_loss,
            "total_loss": total_loss,
            "accuracy": self.accuracy_metric.result(),
        }

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

    def fit(self, initial_epoch, epoches, train_ds, **kwargs):
        history = {
            "train_loss": [],
            "reg_loss": [],
            "b_loss": [],
            "total_loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        global_step = tf.compat.v1.train.get_or_create_global_step()
        global_step.assign(1)
        target = train_ds.size // self.batch_size
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

        for epoch in range(initial_epoch, epoches + 1):
            print("\nEpoch: {}/{}".format(epoch, epoches))

            self.reset_metric()
            seen = 0

            w = tf.summary.create_file_writer(self.summary_path)
            with w.as_default():
                for inputs, labels in train_ds:
                    if inputs.shape.as_list()[0] == self.batch_size:
                        logs = self.train_step(inputs, labels)
                        logs = copy.copy(logs) if logs else {}
                        num_steps = logs.pop("num_steps", 1)
                        seen += num_steps
                        progbar.update(seen, list(logs.items()), finalize=finalize)

                    history["train_loss"].append(self.train_loss_metric.result().numpy())
                    history["reg_loss"].append(self.regularization_loss_metric.result().numpy())
                    history["b_loss"].append(self.boundary_loss_metric.result().numpy())
                    history["total_loss"].append(self.total_loss_metric.result().numpy())
                    history["accuracy"].append(self.accuracy_metric.result().numpy())
                    global_step = tf.compat.v1.train.get_global_step()
                    global_step.assign(global_step + 1)

            self.model.save_model_variables()

        return {"history": history}

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
        print("loss: {:.3f}, accuracy: {:.3f}\n".format(logs["loss"], logs["accuracy"]))
