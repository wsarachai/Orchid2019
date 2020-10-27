from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from pickle import dump

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from data import orchids52_dataset, data_utils
from data.create_orchids_dataset import create_dataset
from data.data_utils import dataset_mapping
from data.orchids52_dataset import TRAIN_SIZE, VALIDATE_SIZE, TEST_SIZE
from lib_utils import latest_checkpoint, start
from nets import nets_utils
from nets.nets_utils import TRAIN_STEP1

flags = tf.compat.v1.flags
logging = tf.compat.v1.logging

FLAGS = flags.FLAGS

flags.DEFINE_string('images_dir', '/Volumes/Data/_dataset/_orchids_dataset/orchids52_data/all',
                    'Original orchid flower images directory')

flags.DEFINE_string('output_directory', '/Volumes/Data/_dataset/_orchids_dataset/orchids52_data/tf-records',
                    'Output data directory')

flags.DEFINE_string('tf_record_dir', '/Volumes/Data/_dataset/_orchids_dataset/orchids52_data/tf-records',
                    'TF record data directory')

flags.DEFINE_string('checkpoint_path', '/Volumes/Data/tmp/orchids-models/orchid2019',
                    'The checkpoint path')

flags.DEFINE_boolean('exp_decay', False,
                     'Exponential decay learning rate')

flags.DEFINE_string('training_step', TRAIN_STEP1,
                    'The training step')

batch_size = 1 
total_epochs = 100


def _main(unused_argv):
    create_dataset(images_dir=FLAGS.images_dir,
                   output_directory=FLAGS.output_directory)


def main(unused_argv):
    load_dataset = dataset_mapping[data_utils.MOBILENET_V2_TFRECORD].load_dataset
    train_ds = load_dataset(
        split="train",
        batch_size=batch_size).repeat()
    validate_ds = load_dataset(
        split="validate",
        batch_size=batch_size).repeat()
    test_ds = load_dataset(
        split="test",
        batch_size=batch_size)

    create_model = nets_utils.nets_mapping[nets_utils.MOBILENET_V2_140_ORCHIDS52]
    model = create_model(num_classes=orchids52_dataset.NUM_OF_CLASSES,
                         is_training=True,
                         batch_size=batch_size,
                         step=FLAGS.training_step)

    if FLAGS.exp_decay:
        base_learning_rate = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=10,
            decay_rate=0.96
        )
    else:
        base_learning_rate = 0.001
        if FLAGS.training_step == 'all':
            base_learning_rate = 0.00001

    optimizer = keras.optimizers.RMSprop(learning_rate=base_learning_rate)

    model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=optimizer,
                  metrics=['accuracy'])

    epochs = 0

    checkpoint_path = os.path.join(FLAGS.checkpoint_path, FLAGS.training_step)
    checkpoint_file = os.path.join(checkpoint_path, 'cp-{epoch:04d}.h5')

    latest, step = latest_checkpoint(FLAGS.training_step)
    if latest:
        epochs = step
        model.load_weights(latest, by_name=True, skip_mismatch=True)
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

    model.summary()

    summary = model.fit(train_ds,
                        epochs=total_epochs,
                        validation_data=validate_ds,
                        callbacks=[cp_callback],
                        initial_epoch=epochs,
                        steps_per_epoch=train_step,
                        validation_steps=validate_step)

    with open('trainHistoryOld', 'wb') as handle:  # saving the history of the model
        dump(summary.history, handle)

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
    start(main)
