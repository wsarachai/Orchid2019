from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from data import orchids52_dataset, data_utils
from data.create_orchids_dataset import create_dataset
from data.data_utils import dataset_mapping
from data.orchids52_dataset import TRAIN_SIZE, VALIDATE_SIZE, TEST_SIZE
from lib_utils import latest_checkpoint
from nets import nets_utils

flags = tf.compat.v1.flags
logging = tf.compat.v1.logging

FLAGS = flags.FLAGS

flags.DEFINE_string('images_dir', '/Volumes/Data/_dataset/_orchids_dataset/orchids52_data/all',
                    'Original orchid flower images directory')

flags.DEFINE_string('output_directory', '/Volumes/Data/_dataset/_orchids_dataset/orchids52_data/tf-records',
                    'Output data directory')

flags.DEFINE_string('tf_record_dir', '/Volumes/Data/_dataset/_orchids_dataset/orchids52_data/tf-records',
                    'TF record data directory')

flags.DEFINE_boolean('exp_decay', False,
                     'Exponential decay learning rate')

batch_size = 1 
total_epochs = 100


def _main(unused_argv):
    create_dataset(images_dir=FLAGS.images_dir,
                   output_directory=FLAGS.output_directory)


def main(unused_argv):
    load_dataset = dataset_mapping[data_utils.MOBILENET_V2_TFRECORD].load_dataset
    train_ds = load_dataset(
        split="train",
        batch_size=batch_size)
    test_ds = load_dataset(
        split="test",
        batch_size=batch_size)
    validate_ds = load_dataset(
        split="validate",
        batch_size=batch_size)

    create_model = nets_utils.nets_mapping[nets_utils.MOBILENET_V2_140_ORCHIDS52]
    model = create_model(num_classes=orchids52_dataset.NUM_OF_CLASSES,
                         freeze_base_model=True,
                         is_training=True,
                         batch_size=batch_size,
                         step='pretrain1')

    if FLAGS.exp_decay:
        base_learning_rate = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.96
        )
    else:
        base_learning_rate = 0.001

    optimizer = keras.optimizers.RMSprop(learning_rate=base_learning_rate)

    model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=optimizer,
                  metrics=['accuracy'])

    checkpoint_path = "/Volumes/Data/tmp/orchids-models/orchid2019"
    checkpoint_file = "/Volumes/Data/tmp/orchids-models/orchid2019/cp-{epoch:04d}.h5"

    epochs = 0

    latest, step = latest_checkpoint(checkpoint_path)
    if latest:
        epochs = step
        chk_file = checkpoint_file.format(epoch=step)
        model.load_weights(chk_file, by_name=True)
    else:
        if not tf.io.gfile.exists(checkpoint_path):
            tf.io.gfile.mkdir(checkpoint_path)
        model.save_weights(checkpoint_file.format(epoch=0))

    # Create a callback that saves the model's weights
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_file,
                                                  save_weights_only=True,
                                                  verbose=1)

    train_step = TRAIN_SIZE // batch_size
    validate_step = VALIDATE_SIZE // batch_size
    test_step = TEST_SIZE // batch_size

    summary = model.fit(train_ds,
                        epochs=total_epochs,
                        validation_data=validate_ds,
                        callbacks=[cp_callback],
                        initial_epoch=epochs,
                        steps_per_epoch=train_step,
                        validation_steps=validate_step)

    if hasattr(summary.history, 'history'):
        acc = summary.history['accuracy']
        val_acc = summary.history['val_accuracy']

        loss = summary.history['loss']
        val_loss = summary.history['val_loss']

        epochs_range = range(total_epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    loss, accuracy = model.evaluate(test_ds, steps=test_step)
    print('Test accuracy :', accuracy)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    logging.info("tf.version %s" % tf.version.VERSION)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.compat.v1.app.run(main)
