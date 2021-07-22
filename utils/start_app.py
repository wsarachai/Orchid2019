from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from absl import app
from absl import logging
from absl import flags
from data.data_utils import DATASET_VERSION_V1, DATA_FORMAT_H5
from data.data_utils import ORCHIDS52
from utils.const import MOBILENET_V2_140_ORCHIDS52

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
    None,
    #"/home/keng/Documents/_trained_models/model-v1/mobilenet_v2_140_orchids52_0001/pretrain2/model.ckpt-12000",
    "Checkpoint Path",
)


def start(start_fn, **kwargs):
    logging.set_verbosity(logging.INFO)
    logging.info("tf.version %s" % tf.version.VERSION)
    app.run(start_fn, **kwargs)
