from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import collections
import numpy as np
import tensorflow as tf
import lib_utils

from absl import flags
from absl import logging
from data import data_utils
from nets import utils
from nets.mobilenet_v2_140 import preprocess_input


flags.DEFINE_boolean("bash", False, "Execute from bash")

flags.DEFINE_boolean("exp_decay", False, "Exponential decay learning rate")

flags.DEFINE_integer("batch_size", 32, "Batch size")

flags.DEFINE_integer("train_step", 1, "Training step")

flags.DEFINE_float("learning_rate", 0.001, "Learning Rate")

flags.DEFINE_string("dataset", data_utils.ORCHIDS52_V1_TFRECORD, "Dataset")

flags.DEFINE_string("model", utils.MOBILENET_V2_140_ORCHIDS52, "Model")

flags.DEFINE_string("checkpoint_path", None, "Checkpoint path")

flags.DEFINE_string(
    "image_dir",
    "/Users/watcharinsarachai/Documents/_datasets/orchids52_data/test/",
    "The directory where the dataset images are locate",
)

FLAGS = flags.FLAGS

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


def main(unused_argv):
    logging.debug(unused_argv)
    dataset_images = create_image_lists(image_dir=FLAGS.image_dir)
    load_dataset = data_utils.dataset_mapping[FLAGS.dataset]
    model = tf.keras.models.load_model(FLAGS.checkpoint_path)
    model.summary()

    count = 0
    corrected = 0
    total_images = load_dataset.test_size
    for label, data in dataset_images.items():
        for file in data["testing"]:
            filename = os.path.join(FLAGS.image_dir, data["dir"], file)
            image_data = tf.io.gfile.GFile(filename, "rb").read()
            inputs = preprocess_input(image_data)
            result = model(inputs).numpy()

            count += 1
            predict = np.argmax(result, axis=1)[0]
            confi = result[0][predict]
            spredict = "n{:04d}".format(predict)
            if spredict == label:
                corrected += 1
            sys.stdout.write(
                "\r>> {}/{}: Predict: {}, expected: {}, confident: {:.4f}, acc: {:.4f}".format(
                    count, total_images, spredict, label, confi, corrected / count
                )
            )
            sys.stdout.flush()

    sys.stdout.write("\n\nDone evaluation -- epoch limit reached")
    sys.stdout.write("Accuracy: {:.4f}".format(corrected / total_images))
    sys.stdout.flush()


if __name__ == "__main__":
    # tf.config.run_functions_eagerly(True)
    lib_utils.start(main)
