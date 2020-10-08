from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import PIL
import PIL.Image
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model


if __name__ == '__main__':
  data_dir = pathlib.Path("/Volumes/Data/_dataset/_orchids_dataset/orchids52_data/all")
  image_count = len(list(data_dir.glob('*/*.jpg')))
  list_ds = tf.data.Dataset.list_files(str(data_dir / '*/*'), shuffle=False)
  list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

  class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != ".DS_Store"]))
  print(class_names)

  val_size = int(image_count * 0.2)
  train_ds = list_ds.skip(val_size)
  val_ds = list_ds.take(val_size)
  val_batches = tf.data.experimental.cardinality(val_ds)
  test_ds = val_ds.take(val_batches // 5)
  val_ds = val_ds.skip(val_batches // 5)

  print(tf.data.experimental.cardinality(train_ds).numpy())
  print(tf.data.experimental.cardinality(val_ds).numpy())
  print(tf.data.experimental.cardinality(test_ds).numpy())

  AUTOTUNE = tf.data.experimental.AUTOTUNE
  num_classes = len(class_names)
  batch_size = 32
  IMG_SIZE = (224, 224)

  def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    one_hot = parts[-2] == class_names
    # Integer encode the label
    return tf.argmax(one_hot)


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


  def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=500)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

  # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
  train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
  val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

  for image, label in train_ds.take(1):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())

  train_ds = configure_for_performance(train_ds)
  val_ds = configure_for_performance(val_ds)

  # Create the base model from the pre-trained model MobileNet V2
  IMG_SHAPE = IMG_SIZE + (3,)
  base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                 include_top=False,
                                                 weights='imagenet')

  image_batch, label_batch = next(iter(train_ds))
  feature_batch = base_model(image_batch)
  print(feature_batch.shape)

  base_model.trainable = False
  base_model.summary()

  global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
  feature_batch_average = global_average_layer(feature_batch)

  prediction_layer = tf.keras.layers.Dense(1)
  prediction_batch = prediction_layer(feature_batch_average)

  data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
  ])

  preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
  #rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1)

  inputs = tf.keras.Input(shape=IMG_SHAPE)
  x = data_augmentation(inputs)
  x = preprocess_input(x)
  x = base_model(x, training=False)
  x = global_average_layer(x)
  x = tf.keras.layers.Dropout(0.2)(x)
  outputs = prediction_layer(x)
  model = tf.keras.Model(inputs, outputs)

  base_learning_rate = 0.0001
  model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])

  model.summary()
  initial_epochs = 10

  loss0, accuracy0 = model.evaluate(val_ds)
  print("initial loss: {:.2f}".format(loss0))
  print("initial accuracy: {:.2f}".format(accuracy0))

  history = model.fit(train_ds,
                      epochs=initial_epochs,
                      validation_data=val_ds)

  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  plt.figure(figsize=(8, 8))
  plt.subplot(2, 1, 1)
  plt.plot(acc, label='Training Accuracy')
  plt.plot(val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.ylabel('Accuracy')
  plt.ylim([min(plt.ylim()), 1])
  plt.title('Training and Validation Accuracy')

  plt.subplot(2, 1, 2)
  plt.plot(loss, label='Training Loss')
  plt.plot(val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.ylabel('Cross Entropy')
  plt.ylim([0, 1.0])
  plt.title('Training and Validation Loss')
  plt.xlabel('epoch')
  plt.show()

  base_model.trainable = True

  # Let's take a look to see how many layers are in the base model
  print("Number of layers in the base model: ", len(base_model.layers))

  # Fine-tune from this layer onwards
  fine_tune_at = 100

  # Freeze all the layers before the `fine_tune_at` layer
  for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

  model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate / 10),
                metrics=['accuracy'])

  model.summary()
  len(model.trainable_variables)

  fine_tune_epochs = 10
  total_epochs = initial_epochs + fine_tune_epochs

  history_fine = model.fit(train_ds,
                           epochs=total_epochs,
                           initial_epoch=history.epoch[-1],
                           validation_data=val_ds)

  acc += history_fine.history['accuracy']
  val_acc += history_fine.history['val_accuracy']

  loss += history_fine.history['loss']
  val_loss += history_fine.history['val_loss']

  plt.figure(figsize=(8, 8))
  plt.subplot(2, 1, 1)
  plt.plot(acc, label='Training Accuracy')
  plt.plot(val_acc, label='Validation Accuracy')
  plt.ylim([0.8, 1])
  plt.plot([initial_epochs - 1, initial_epochs - 1],
           plt.ylim(), label='Start Fine Tuning')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(2, 1, 2)
  plt.plot(loss, label='Training Loss')
  plt.plot(val_loss, label='Validation Loss')
  plt.ylim([0, 1.0])
  plt.plot([initial_epochs - 1, initial_epochs - 1],
           plt.ylim(), label='Start Fine Tuning')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.xlabel('epoch')
  plt.show()

  loss, accuracy = model.evaluate(test_ds)
  print('Test accuracy :', accuracy)

