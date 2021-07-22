from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from data import data_utils
from data.data_utils import dataset_mapping
from utils.start_app import start

flags = tf.compat.v1.flags
logging = tf.compat.v1.logging
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "tf_record_dir", "/Volumes/Data/_dataset/_orchids_dataset/orchids52_data/v1/tf-records", "TF record data directory"
)

flags.DEFINE_boolean("exp_decay", False, "Exponential decay learning rate")

flags.DEFINE_integer("batch_size", 32, "Batch size")

flags.DEFINE_integer("total_epochs", 100, "Total epochs")

flags.DEFINE_integer("start_state", 1, "Start state")

flags.DEFINE_integer("end_state", 5, "End state")

flags.DEFINE_float("learning_rate", 0.001, "Learning Rate")

flags.DEFINE_string("aug_method", "fast", "Augmentation Method")


def main(unused_argv):
    logging.debug(unused_argv)
    num_classes = 52
    workspace_path = os.environ["WORKSPACE"] or "/Volumes/Data/tmp"
    data_path = os.environ["DATA_DIR"] or "/Volumes/Data/_dataset/_orchids_dataset"
    data_dir = os.path.join(data_path, "orchids52_data")
    load_dataset = dataset_mapping[data_utils.ORCHIDS52_V2_FILE]

    train_ds = load_dataset(
        split="train", batch_size=FLAGS.batch_size, root_path=data_dir, aug_method=FLAGS.aug_method
    )
    validate_ds = load_dataset(split="validate", batch_size=FLAGS.batch_size, root_path=data_dir)
    test_ds = load_dataset(split="test", batch_size=FLAGS.batch_size, root_path=data_dir)

    # Create the base model from the pre-trained model MobileNet V2
    IMG_SHAPE = (224, 224, 3)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE, alpha=1.4, include_top=False, weights="imagenet"
    )

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(num_classes, activation=tf.keras.activations.linear)

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        ]
    )

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
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
        metrics=["accuracy"],
    )

    model.summary()
    total_epochs = 50

    model.fit(train_ds, epochs=total_epochs, validation_data=validate_ds)

    loss, accuracy = model.evaluate(test_ds)
    print("Test accuracy :", accuracy)


if __name__ == "__main__":
    tf.config.experimental_run_functions_eagerly(True)
    start(main)
