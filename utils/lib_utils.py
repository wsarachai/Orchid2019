from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import re
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

flags.DEFINE_integer("batch_size", 32, "Batch size")

flags.DEFINE_integer("train_step", 1, "Training step")

flags.DEFINE_float("learning_rate", 0.001, "Learning Rate")

flags.DEFINE_string("dataset_format", DATA_FORMAT_H5, "Dataset format")

flags.DEFINE_string("dataset", ORCHIDS52, "Dataset")

flags.DEFINE_string("dataset_version", DATASET_VERSION_V1, "Dataset version")

flags.DEFINE_string("model", MOBILENET_V2_140_ORCHIDS52, "Model")

flags.DEFINE_string("checkpoint_dir", "mobilenet_v2_140_orchids52_0001", "Checkpoint directory")

flags.DEFINE_string(
    "learning_rate_decay", "", "Exponential decay learning rate, exponential, cosine, piecewise_onstant"
)

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
    "trained_dir",
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


def config_learning_rate(learning_rate=0.001, decay=""):
    if decay == "exponential":
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate, decay_steps=10, decay_rate=0.96
        )
    elif decay == "cosine":
        learning_rate = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=learning_rate, decay_steps=710)
    elif decay == "piecewise_constant":
        boundaries = [700, 1400]
        values = [learning_rate, learning_rate / 5, learning_rate / 10]
        learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

    return learning_rate


def config_optimizer(optimizer, learning_rate):
    if optimizer == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        return tf.keras.optimizers.SGD(learning_rate=learning_rate)


def config_loss(**kwargs):
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    return loss_fn


class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate):
        self.initial_learning_rate = initial_learning_rate

    def __call__(self, step):
        return self.initial_learning_rate / (step + 1)


class DisplayInfo(object):
    def __init__(self, total_images):
        self.count = 0
        self.corrected = 0
        self.total_images = total_images

    def display_info(self, result, label, count):
        predict = np.argmax(result, axis=1)[0]
        confident = result[0][predict]
        try:
            predict_string = "n{:04d}".format(predict)
            if label.dtype != tf.string:
                label = "n{:04d}".format(label)
            if predict_string == label:
                self.corrected += 1
        except:
            try:
                label = np.argmax(label, axis=1)[0]
                if predict == label:
                    self.corrected += 1
            except:
                if predict == label:
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
