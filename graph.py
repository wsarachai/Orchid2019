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
    logging.debug(unused_argv)
    file_to_load = '{}.pack'.format(FLAGS.file)
    history = load(open(file_to_load, 'rb'))

    train_loss = history['train_loss']
    regularization_loss = history['regularization_loss']
    boundary_loss = history['boundary_loss']
    total_loss = history['total_loss']
    accuracy = history['accuracy']
    validation_loss = history['validation_loss']
    validation_accuracy = history['validation_accuracy']
    num_of_data = len(accuracy)
    epochs_range = range(num_of_data)

    fig, axs = plt.subplots(3, 1, figsize=(10, 10), dpi=300)

    axs[0].plot(epochs_range, accuracy, label='Training Accuracy')
    axs[0].plot(epochs_range, validation_accuracy, label='Validation Accuracy')
    axs[0].set_xlim(0, num_of_data)
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend(loc='lower right')
    axs[0].grid(True)

    axs[1].plot(epochs_range, train_loss, label='Training Loss')
    axs[1].plot(epochs_range, validation_loss, label='Validation Loss')
    axs[1].set_xlim(0, num_of_data)
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].legend(loc='upper right')
    axs[1].grid(True)

    axs[2].plot(epochs_range, regularization_loss, label='Regularization Loss')
    axs[2].plot(epochs_range, boundary_loss, label='Boundary Loss')
    axs[2].plot(epochs_range, total_loss, label='Total Loss')
    axs[2].set_xlim(0, num_of_data)
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Loss')
    axs[2].legend(loc='center right')
    axs[2].grid(True)

    fig.tight_layout()
    plt.savefig('{}.png'.format(FLAGS.file))


if __name__ == '__main__':
    start(main)
