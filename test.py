from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from data import data_utils, orchids52_dataset
from data.data_utils import dataset_mapping
from data.orchids52_dataset import TEST_SIZE
from lib_utils import latest_checkpoint, start
from nets import nets_utils
from nets.nets_utils import TRAIN_STEP1

flags = tf.compat.v1.flags
logging = tf.compat.v1.logging
FLAGS = flags.FLAGS

flags.DEFINE_string('training_step', TRAIN_STEP1,
                    'The training step')

batch_size = 1


def main(unused_argv):
    load_dataset = dataset_mapping[data_utils.MOBILENET_V2_TFRECORD].load_dataset
    test_ds = load_dataset(
        split="test",
        batch_size=batch_size)

    create_model = nets_utils.nets_mapping[nets_utils.MOBILENET_V2_140_ORCHIDS52]
    model = create_model(num_classes=orchids52_dataset.NUM_OF_CLASSES,
                         is_training=True,
                         batch_size=batch_size,
                         step=FLAGS.training_step)

    model.compile(metrics=['accuracy'])

    latest, step = latest_checkpoint(train_step=FLAGS.training_step)
    if latest:
        model.load_weights(str(latest), by_name=True)
        test_step = TEST_SIZE // batch_size
        loss, accuracy = model.evaluate(test_ds, steps=test_step)
        print('Test accuracy :', accuracy)


if __name__ == '__main__':
    start(main)
