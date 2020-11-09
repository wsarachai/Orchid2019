from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import matplotlib.pyplot as plt
from data import data_utils
from data.data_utils import dataset_mapping
from lib_utils import start
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from nets.mobilenet_v2 import create_mobilenet_v2, IMG_SHAPE_224

logging = tf.compat.v1.logging


def main(unused_argv):
    logging.debug(unused_argv)
    batch_size = 32
    data_path = os.environ['DATA_DIR'] or '/Volumes/Data/_dataset/_orchids_dataset'
    data_dir = os.path.join(data_path, 'orchids52_data')
    load_dataset = dataset_mapping[data_utils.ORCHIDS52_V1_TFRECORD]
    train_ds = load_dataset(split="train", batch_size=batch_size, root_path=data_dir)
    validate_ds = load_dataset(split="validate", batch_size=batch_size, root_path=data_dir)

    train_step = train_ds.size // batch_size
    validate_step = validate_ds.size // batch_size

    data_augmentation = Sequential([
            layers.experimental.preprocessing.RandomCrop(200, 200),
            layers.experimental.preprocessing.RandomFlip("horizontal",input_shape=(224, 224, 3)),
            layers.experimental.preprocessing.RandomRotation(0.1)
        ])

    # for images, labels in train_ds.take(1):
    #     for i in range(9):
    #         augmented_images = data_augmentation(images)
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.imshow(augmented_images[i].numpy().astype("uint8"))
    #         plt.title(str(i))
    #         plt.axis("off")
    #     plt.show()

    # model = Sequential([
    #     data_augmentation,
    #     layers.experimental.preprocessing.Rescaling(1. / 255),
    #     layers.Conv2D(16, 3, padding='same', activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Conv2D(32, 3, padding='same', activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Conv2D(64, 3, padding='same', activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Dropout(0.2),
    #     layers.Flatten(),
    #     layers.Dense(128, activation='relu'),
    #     layers.Dense(52)
    # ])

    inputs = layers.Input(shape=(224, 224, 3), batch_size=batch_size)

    base_model = create_mobilenet_v2(input_tensor=inputs,
                                     alpha=1.4,
                                     include_top=False,
                                     weights='imagenet',
                                     sub_name='02')

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(52)(x)
    model = tf.keras.Model(inputs, outputs)

    # optimizer = keras.optimizers.RMSprop()
    # model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    #               optimizer=optimizer,
    #               metrics=['accuracy'])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    base_model.trainable = False

    model.summary()

    # for inp, labels in train_ds:
    #     result = model(inp)
    #     print(result)

    summary = model.fit(train_ds,
                        epochs=100,
                        validation_data=validate_ds,
                        validation_steps=validate_step,
                        initial_epoch=1,
                        steps_per_epoch=train_step)

    img, lbl = next(iter(train_ds))
    print(lbl)


if __name__ == '__main__':
    tf.config.experimental_run_functions_eagerly(True)
    start(main)