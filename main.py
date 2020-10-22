from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tensorflow import keras
from data.orchids import load_dataset
from data.orchids import IMG_SIZE_224
from lib_utils import latest_checkpoint, create_orchid_mobilenet_v2_14_cus

flags = tf.compat.v1.flags
logging = tf.compat.v1.logging

FLAGS = flags.FLAGS

flags.DEFINE_string('images_dir', '/Volumes/Data/_dataset/_orchids_dataset/orchids52_data/all',
                    'Original orchid flower images directory')

flags.DEFINE_string('output_directory', '/Volumes/Data/_dataset/_orchids_dataset/orchids52_data/tf-records',
                    'Output data directory')

flags.DEFINE_string('tf_record_dir', '/Volumes/Data/_dataset/_orchids_dataset/orchids52_data/tf-records',
                    'TF record data directory')

num_classes = 52
batch_size = 32


def decode_img(image, size):
    img = tf.image.decode_jpeg(image, channels=3)
    return tf.image.resize(img, size)


def get_label(serialize_example):
    label = serialize_example['image/class/label']
    label_string = tf.strings.split(label, ',')
    label_values = tf.strings.to_number(label_string, out_type=tf.dtypes.int64)
    return label_values


def get_ds(serialize_example):
    image = serialize_example['image/image_raw']
    image = decode_img(image=image, size=IMG_SIZE_224)
    label_values = get_label(serialize_example)
    return image, label_values


def get_dataset(filename):
    dataset = load_dataset(filename)
    dataset = dataset.map(get_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(1000)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size=batch_size)
    return dataset


def main(unused_argv):
    train_file = os.path.join(FLAGS.tf_record_dir, "orchids52-train.tfrecord")
    test_file = os.path.join(FLAGS.tf_record_dir, "orchids52-test.tfrecord")
    validate_file = os.path.join(FLAGS.tf_record_dir, "orchids52-validate.tfrecord")

    train_ds = get_dataset(train_file)
    test_ds = get_dataset(test_file)
    validate_ds = get_dataset(validate_file)

    model = create_orchid_mobilenet_v2_14_cus(num_classes=num_classes,
                                              freeze_base_model=True,
                                              is_training=True,
                                              batch_size=batch_size,
                                              step='pretrain1')

    exp_decay = False
    if exp_decay:
        base_learning_rate = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.96
        )
    else:
        base_learning_rate = 0.001

    optimizer = keras.optimizers.RMSprop(learning_rate=base_learning_rate)

    model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=optimizer,
                  metrics=['accuracy'])

    checkpoint_path = "/Volumes/Data/tmp/orchids-models/orchid2019"
    checkpoint_file = "/Volumes/Data/tmp/orchids-models/orchid2019/cp-{epoch:04d}.h5"

    epochs = 0
    total_epochs = 100

    latest, step = latest_checkpoint(checkpoint_path)
    if latest:
        checkpoint_file = checkpoint_file.format(epoch=step)
        model.load_weights(checkpoint_file, by_name=True)
    else:
        if not tf.io.gfile.exists(checkpoint_path):
            tf.io.gfile.mkdir(checkpoint_path)
        model.save_weights(checkpoint_file.format(epoch=0))

    # Create a callback that saves the model's weights
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_file,
                                                  save_weights_only=True,
                                                  verbose=1)

    history_fine = model.fit(train_ds,
                             epochs=total_epochs,
                             validation_data=validate_ds,
                             callbacks=[cp_callback],
                             initial_epoch=epochs,
                             verbose=1)

    loss, accuracy = model.evaluate(test_ds, verbose=2)
    print('Test accuracy :', accuracy)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    logging.info("tf.version %s" % tf.version.VERSION)
    tf.compat.v1.app.run(main)
