from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import re
import copy
import sys
import numpy as np
import tensorflow as tf

from absl import app
from absl import logging
from absl import flags
from data.data_utils import DATASET_VERSION_V1, DATA_FORMAT_H5
from data.data_utils import ORCHIDS52
from utils.const import MOBILENET_V2_140_ORCHIDS52
from utils.const import TRAIN_V2_STEP2, TRAIN_STEP4

FLAGS = flags.FLAGS

flags.DEFINE_boolean("bash", False, "Execute from bash")

flags.DEFINE_boolean("exp_decay", False, "Exponential decay learning rate")

flags.DEFINE_integer("batch_size", 32, "Batch size")

flags.DEFINE_integer("train_step", 1, "Training step")

flags.DEFINE_float("learning_rate", 0.001, "Learning Rate")

flags.DEFINE_string("dataset_format", DATA_FORMAT_H5, "Dataset format")

flags.DEFINE_string("dataset", ORCHIDS52, "Dataset")

flags.DEFINE_string("dataset_version", DATASET_VERSION_V1, "Dataset version")

flags.DEFINE_string("model", MOBILENET_V2_140_ORCHIDS52, "Model")

flags.DEFINE_string("checkpoint_path", "mobilenet_v2_140_orchids52_0001", "Checkpoint path")

flags.DEFINE_boolean("save_best_only", False, "Save the checkpoint only best result.")

flags.DEFINE_boolean("save_model", False, "Save the model on each state.")

flags.DEFINE_integer("total_epochs", 1, "Total epochs")

flags.DEFINE_integer("start_state", 1, "Start state")

flags.DEFINE_integer("end_state", 2, "End state")

flags.DEFINE_string("file", "trainHistory", "Train history")


flags.DEFINE_string(
    "optimizer",
    "rmsprop",
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",' '"ftrl", "momentum", "sgd" or "rmsprop".',
)

flags.DEFINE_string(
    "trained_path",
    "/home/keng/Documents/_trained_models/model-v1/mobilenet_v2_140_orchids52_0001/pretrain2/model.ckpt-12000",
    "Checkpoint Path",
)


def create_image_lists(image_dir):
    if not tf.io.gfile.exists(image_dir):
        logging.error("Image directory '" + image_dir + "' not found.")
        return None

    result = collections.OrderedDict()
    sub_dirs = sorted(x[0] for x in tf.io.gfile.walk(image_dir))

    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        extensions = sorted(
            set(os.path.normcase(ext) for ext in ["JPEG", "JPG", "jpeg", "jpg", "png"])  # Smash case on Windows.
        )

        file_list = []
        dir_name = os.path.basename(sub_dir[:-1] if sub_dir.endswith("/") else sub_dir)

        if dir_name == image_dir:
            continue
        logging.info("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, "*." + extension)
            file_list.extend(tf.io.gfile.glob(file_glob))
        if not file_list:
            logging.warning("No files found")
            continue

        label_name = re.sub(r"[^a-z0-9]+", " ", dir_name.lower())

        testing_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            testing_images.append(base_name)

        result[label_name] = {"dir": dir_name, "testing": testing_images}

    return result


def get_checkpoint_file(checkpoint_dir, name):
    if isinstance(name, int):
        return os.path.join(checkpoint_dir, "cp-{name:04d}".format(name=name))
    else:
        return os.path.join(checkpoint_dir, "cp-{name}".format(name=name))


def get_step_number(checkpoint_dir):
    idx = checkpoint_dir.index(".")
    step = int(checkpoint_dir[:idx][-4:])
    return step


def apply_with_random_selector(x, func, num_cases):
    sel = tf.random.uniform([], maxval=num_cases, dtype=tf.int32)
    return tf.raw_ops.Merge(
        inputs=[func(tf.raw_ops.Switch(data=x, pred=tf.equal(sel, case))[1], case) for case in range(num_cases)]
    )[0]


def start(start_fn, **kwargs):
    logging.set_verbosity(logging.INFO)
    logging.info("tf.version %s" % tf.version.VERSION)
    app.run(start_fn, **kwargs)


def config_learning_rate(learning_rate=0.001, exp_decay=False, **kwargs):
    training_step = kwargs.pop("training_step") if "training_step" in kwargs else ""
    if exp_decay:
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate, decay_steps=10, decay_rate=0.96
        )
    else:
        if training_step in [TRAIN_STEP4, TRAIN_V2_STEP2]:
            learning_rate = 0.000001
    return learning_rate


def config_optimizer(optimizer, learning_rate, **kwargs):
    if optimizer == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        return tf.keras.optimizers.SGD(learning_rate=learning_rate)


def config_loss(**kwargs):
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    return loss_fn


class TrainClassifier:
    def __init__(self, model, batch_size):
        self.model = model
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


class DisplayInfo(object):
    def __init__(self, total_images):
        self.count = 0
        self.corrected = 0
        self.total_images = total_images

    def display_info(self, result, label, count):
        predict = np.argmax(result, axis=1)[0]
        confident = result[0][predict]
        predict_string = "n{:04d}".format(predict)
        if label.dtype != tf.string:
            label = "n{:04d}".format(label[0])
        if predict_string == label:
            self.corrected += 1
        sys.stdout.write(
            "\r>> {}/{}: Predict: {}, expected: {}, confident: {:.4f}, acc: {:.4f}".format(
                count, self.total_images, predict_string, label, confident, self.corrected / count
            )
        )
        sys.stdout.flush()

    def display_summary(self):
        sys.stdout.write("\n\nDone evaluation -- epoch limit reached")
        sys.stdout.write("Accuracy: {:.4f}".format(self.corrected / self.total_images))
        sys.stdout.flush()
