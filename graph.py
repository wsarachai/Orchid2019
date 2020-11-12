from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt
from pickle import load

from lib_utils import start

flags = tf.compat.v1.flags
logging = tf.compat.v1.logging
FLAGS = flags.FLAGS

flags.DEFINE_string('file', 'trainHistory',
                    'Train history')

flags.DEFINE_integer('total_epochs', 100,
                     'Total epochs')


def main(unused_argv):
    file_to_load = '{}.pack'.format(FLAGS.file)
    history = load(open(file_to_load, 'rb'))

    train_loss = history['train_loss']
    regularization_loss = history['regularization_loss']
    boundary_loss = history['boundary_loss']
    total_loss = history['total_loss']
    accuracy = history['accuracy']
    validation_loss = history['validation_loss']
    validation_accuracy = history['validation_accuracy']
    epochs_range = range(len(accuracy))

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, accuracy, label='Training Accuracy')
    plt.plot(epochs_range, validation_accuracy, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, train_loss, label='Training Loss')
    plt.plot(epochs_range, validation_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('{}.png'.format(FLAGS.file))


if __name__ == '__main__':
    start(main)
