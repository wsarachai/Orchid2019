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
    train_data_dir = pathlib.Path("/Volumes/Data/dataset/_orchids_dataset/orchids52_data/train-en")
    test_data_dir = pathlib.Path("/Volumes/Data/dataset/_orchids_dataset/orchids52_data/test-en")

    image_count = len(list(train_data_dir.glob('*/*.jpg')))
    train_ds = tf.data.Dataset.list_files(str(train_data_dir / '*/*'), shuffle=False)
    train_ds = train_ds.shuffle(image_count, reshuffle_each_iteration=False)
    test_ds = tf.data.Dataset.list_files(str(test_data_dir / '*/*'), shuffle=False)

    class_names = np.array(sorted([item.name for item in train_data_dir.glob('*')]))
    print(class_names)

    val_batches = tf.data.experimental.cardinality(test_ds)
    val_ds = test_ds.skip(val_batches // 5)
    test_ds = test_ds.take(val_batches // 5)

    print(tf.data.experimental.cardinality(train_ds).numpy())
    print(tf.data.experimental.cardinality(val_ds).numpy())
    print(tf.data.experimental.cardinality(test_ds).numpy())

    num_classes = len(class_names)
    IMG_SIZE = (224, 224)

    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    train_ds = train_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    for image, label in train_ds.take(1):
        print("Image shape: ", image.numpy().shape)
        print("Label: ", label.numpy())

    train_ds = configure_for_performance(train_ds)
    val_ds = configure_for_performance(val_ds)
    test_ds = configure_for_performance(test_ds)

    # Create the base model from the pre-trained model MobileNet V2
    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   alpha=1.4,
                                                   include_top=False,
                                                   weights='imagenet')

    image_batch, label_batch = next(iter(train_ds))
    feature_batch = base_model(image_batch)
    print(feature_batch.shape)

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(num_classes)

    feature_batch_average = global_average_layer(feature_batch)
    prediction_batch = prediction_layer(feature_batch_average)

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    # Freeze all the layers except for dense layer
    for layer in base_model.layers:
        layer.trainable = False

    base_learning_rate = 0.0001
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                  metrics=['accuracy'])

    model.summary()
    total_epochs = 50

    history_fine = model.fit(train_ds,
                             epochs=total_epochs,
                             validation_data=val_ds)

    loss, accuracy = model.evaluate(test_ds)
    print('Test accuracy :', accuracy)
