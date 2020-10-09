from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import pathlib
import tensorflow as tf


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    one_hot = parts[-2] == class_names
    return tf.cast(one_hot, tf.float32)


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # resize the image to the desired size
    return tf.image.resize(img, IMG_SIZE)


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def configure_for_performance(ds, batch_size=32):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


if __name__ == '__main__':
    test_data_dir = pathlib.Path("/Volumes/Data/dataset/_orchids_dataset/orchids52_data/test-en")
    test_ds = tf.data.Dataset.list_files(str(test_data_dir / '*/*'), shuffle=False)
    class_names = np.array(sorted([item.name for item in test_data_dir.glob('*')]))
    print(class_names)
    print(tf.data.experimental.cardinality(test_ds).numpy())

    num_classes = len(class_names)
    IMG_SIZE = (224, 224)

    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    test_ds = test_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    for image, label in test_ds.take(1):
        print("Image shape: ", image.numpy().shape)
        print("Label: ", label.numpy())

    test_ds = configure_for_performance(test_ds)

    # Create the base model from the pre-trained model MobileNet V2
    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   alpha=1.4,
                                                   include_top=False,
                                                   weights='imagenet')

    image_batch, label_batch = next(iter(test_ds))
    feature_batch = base_model(image_batch)
    print(feature_batch.shape)

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(num_classes)

    feature_batch_average = global_average_layer(feature_batch)
    prediction_batch = prediction_layer(feature_batch_average)

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    model.summary()

    checkpoint_path = "/Volumes/Data/tmp/orchids-models/orchid2019/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.RMSprop(),
                  metrics=['accuracy'])

    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)

    loss, accuracy = model.evaluate(test_ds)
    print('Test accuracy :', accuracy)
