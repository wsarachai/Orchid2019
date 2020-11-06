from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from data import data_utils
from data.data_utils import dataset_mapping
from lib_utils import start
from nets.mobilenet_v2 import create_mobilenet_v2, IMG_SHAPE_224

logging = tf.compat.v1.logging

def printname(name):
    print(name)


def main(unused_argv):
    logging.debug(unused_argv)
    data_path = os.environ['DATA_DIR'] or '/Volumes/Data/_dataset/_orchids_dataset'
    data_dir = os.path.join(data_path, 'orchids52_data')
    load_dataset = dataset_mapping[data_utils.ORCHIDS52_V1_TFRECORD]
    train_ds = load_dataset(split="train", batch_size=2, root_path=data_dir)

    img, lbl = next(iter(train_ds))

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    inputs = tf.keras.Input(shape=(224, 224, 3), batch_size=1)
    inputs = preprocess_input(inputs)

    base_model = create_mobilenet_v2(input_tensor=inputs,
                                     alpha=1.4,
                                     include_top=False,
                                     weights='imagenet',
                                     sub_name='02')

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(52, activation='relu'),
    ])

    optimizer = tf.keras.optimizers.RMSprop()

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  optimizer=optimizer,
                  metrics=['accuracy'])

    base_model.trainable = False

    model.summary()

    train_step = train_ds.size

    summary = model.fit(train_ds,
                        epochs=100,
                        steps_per_epoch=train_step)

    img, lbl = next(iter(train_ds))
    print(lbl)


if __name__ == '__main__':
    tf.config.experimental_run_functions_eagerly(True)
    start(main)