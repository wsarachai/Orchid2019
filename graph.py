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
    max_val = 0
    max_s = 0
    min_lost = 1
    min_s = 0
    history = load(open(FLAGS.file, 'rb'))

    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs_range = range(len(val_acc))

    for i in epochs_range:
        if val_acc[i] > max_val:
            max_val = val_acc[i]
            max_s = i
        if val_loss[i] < min_lost:
            min_lost = val_loss[i]
            min_s = i

    logging.info('max validate {}: {:.2f}, loss: {:.2f}'.format(max_s, max_val, val_loss[max_s]))
    logging.info('min loss {}: {:.2f}, validate: {:.2f}'.format(min_s, min_lost, val_acc[min_s]))

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
    #plt.show()
    plt.savefig("pretrain2.png")


if __name__ == '__main__':
    start(main)
