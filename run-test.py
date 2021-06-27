from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
import tensorflow as tf
import numpy as np
import h5py
import lib_utils

from lib_utils import FLAGS
from data import data_utils
from nets import utils
from absl import logging


def main(unused_argv):
    logging.debug(unused_argv)
    load_dataset = data_utils.dataset_mapping[FLAGS.dataset]
    create_model = utils.nets_mapping[FLAGS.model]

    model = create_model(num_classes=load_dataset.num_of_classes, optimizer=None, loss_fn=None, batch_size=1, step="")
    accuracy_metric = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")
    model.compile(metrics=[accuracy_metric])
    model.restore_model_variables(checkpoint_path=FLAGS.checkpoint_path, show_model_weights=True)
    #model.summary()

    save_path = FLAGS.image_dir + '-new'
    f = h5py.File(save_path + '/orchids52.h5', 'r')
    dataset = f['orchids52/test']

    info = lib_utils.DisplayInfo(load_dataset.test_size)

    count = 0
    for label in dataset:
        label = 'n0001'
        for inputs in dataset[label]:
            inputs = np.expand_dims(inputs, axis=0)
            result = model.model(inputs).numpy()

            count = count + 1
            info.display_info(result, label, count)

    info.display_summary()


if __name__ == "__main__":
    tf.config.run_functions_eagerly(True)
    lib_utils.start(main)
