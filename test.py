from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow import keras
from data.orchids import create_dataset
from lib_utils import latest_checkpoint, create_orchid_mobilenet_v2_14_cus


if __name__ == '__main__':
    _, _, test_ds, num_classes = create_dataset(2)

    model = create_orchid_mobilenet_v2_14_cus(num_classes=num_classes,
                                              freeze_base_model=True,
                                              is_training=True,
                                              step='pretrain1')

    checkpoint_path = "/Volumes/Data/tmp/orchids-models/orchid2019/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    base_learning_rate = 0.001
    optimizer = keras.optimizers.RMSprop(learning_rate=base_learning_rate)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=optimizer,
                  metrics=['accuracy'])

    latest, step = latest_checkpoint(checkpoint_dir)
    if latest:
        epochs = step
        model.load_weights(str(latest), by_name=True)

        loss, accuracy = model.evaluate(test_ds)
        print('Test accuracy :', accuracy)
