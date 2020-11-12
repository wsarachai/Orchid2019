from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from data import data_utils, orchids52_dataset
from data.data_utils import dataset_mapping
from lib_utils import start
from nets import utils

flags = tf.compat.v1.flags
logging = tf.compat.v1.logging
FLAGS = flags.FLAGS

flags.DEFINE_string('tf_record_dir', '/Volumes/Data/_dataset/_orchids_dataset/orchids52_data/v1/tf-records',
                    'TF record data directory')

flags.DEFINE_boolean('exp_decay', False,
                     'Exponential decay learning rate')

flags.DEFINE_integer('batch_size', 32,
                     'Batch size')

flags.DEFINE_integer('total_epochs', 50,
                     'Total epochs')

flags.DEFINE_integer('start_state', 1,
                     'Start state')

flags.DEFINE_integer('end_state', 5,
                     'End state')

flags.DEFINE_float('learning_rate', 0.001,
                   'Learning Rate')

flags.DEFINE_string('aug_method', 'fast',
                    'Augmentation Method')


def main(unused_argv):
    logging.debug(unused_argv)
    data_path = os.environ['DATA_DIR'] if 'DATA_DIR' in os.environ else '/Volumes/Data/_dataset/_orchids_dataset'
    data_dir = os.path.join(data_path, 'orchids52_data')
    load_dataset = dataset_mapping[data_utils.ORCHIDS52_V1_TFRECORD]
    create_model = utils.nets_mapping[utils.MOBILENET_V2_140]

    train_ds = load_dataset(split="train",
                            batch_size=FLAGS.batch_size,
                            root_path=data_dir,
                            aug_method=FLAGS.aug_method)
    validate_ds = load_dataset(split="validate", batch_size=FLAGS.batch_size, root_path=data_dir)
    test_ds = load_dataset(split="test", batch_size=FLAGS.batch_size, root_path=data_dir)

    # import matplotlib.pyplot as plt
    # for images, _ in train_ds.take(1):
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.imshow(images[i].numpy().astype("uint8"))
    #         plt.title(str(i))
    #         plt.axis("off")
    #     plt.show()

    model = create_model(num_classes=orchids52_dataset.NUM_OF_CLASSES,
                         training=True,
                         batch_size=FLAGS.batch_size)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.RMSprop(lr=FLAGS.learning_rate),
                  metrics=['accuracy'])

    model.summary()
    total_epochs = 50

    history_fine = model.fit(train_ds,
                             epochs=total_epochs,
                             validation_data=validate_ds)

    loss, accuracy = model.evaluate(test_ds)
    print('Test accuracy :', accuracy)


if __name__ == '__main__':
    tf.config.experimental_run_functions_eagerly(True)
    start(main)
