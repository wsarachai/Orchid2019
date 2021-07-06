from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import collections
import h5py
import tensorflow as tf
import numpy as np

from absl import logging
from nets.mobilenet_v2_140 import preprocess_input
from utils.lib_utils import FLAGS


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


def main(_):
    split = "train"
    workspace_path = os.environ["WORKSPACE"] if "WORKSPACE" in os.environ else "/Users/watcharinsarachai/Documents/"
    image_dir = os.path.join(
        workspace_path, "_datasets", FLAGS.dataset, FLAGS.dataset_format, FLAGS.dataset_version, split
    )
    save_path = os.path.join(workspace_path, "_datasets", FLAGS.dataset, "h5", FLAGS.dataset_version, split)
    dataset_images = create_image_lists(image_dir=image_dir)
    if not tf.io.gfile.exists(save_path):
        os.makedirs(save_path)
    f = h5py.File(save_path + "/orchids52.h5", "w")
    for label, data in dataset_images.items():
        print(label, ": ", len(data))
        elems = []
        for file in sorted(data["testing"]):
            filename = os.path.join(image_dir, data["dir"], file)
            image_data = tf.io.gfile.GFile(filename, "rb").read()
            inputs = preprocess_input(image_data)
            elems.append(tf.squeeze(inputs))
        elems = np.stack([e for e in elems])
        print(elems.shape)
        dset = f.create_dataset("orchids52/{}/".format(split) + label, elems.shape)
        for i, e in enumerate(elems):
            dset[i] = e

    # save_path = FLAGS.image_dir + '-new'
    # f = h5py.File(save_path + '/orchids52.h5', 'r')
    # dset = f['orchids52/test']
    # for label in dset:
    #     for d in dset[label]:
    #         print(d.shape)


if __name__ == "__main__":
    # tf.config.run_functions_eagerly(True)
    lib_utils.start(main)
