from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow import keras

from data import data_utils, orchids52_dataset
from data.data_utils import dataset_mapping
from data.orchids52_dataset import TEST_SIZE
from lib_utils import latest_checkpoint, start
from nets import nets_utils

flags = tf.compat.v1.flags
logging = tf.compat.v1.logging

batch_size = 1


def main(unused_argv):
    load_dataset = dataset_mapping[data_utils.MOBILENET_V2_TFRECORD].load_dataset
    test_ds = load_dataset(
        split="test",
        batch_size=batch_size)

    create_model = nets_utils.nets_mapping[nets_utils.MOBILENET_V2_140_ORCHIDS52]
    model = create_model(num_classes=orchids52_dataset.NUM_OF_CLASSES,
                         freeze_base_model=True,
                         is_training=True,
                         batch_size=batch_size,
                         step='pretrain1')

    checkpoint_path = "/Volumes/Data/tmp/orchids-models/orchid2019/cp-{epoch:04d}.h5"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    model.compile(metrics=['accuracy'])

    latest, step = latest_checkpoint(checkpoint_dir)
    if latest:
        model.load_weights(str(latest), by_name=True)
        test_step = TEST_SIZE // batch_size
        loss, accuracy = model.evaluate(test_ds, steps=test_step)
        print('Test accuracy :', accuracy)


if __name__ == '__main__':
    start(main)
