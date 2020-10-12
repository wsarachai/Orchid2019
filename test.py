from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

import lib_utils


if __name__ == '__main__':
    _, _, test_ds, num_classes = lib_utils.create_dataset()

    model = lib_utils.create_orchid_mobilenet_v2_14(num_classes=num_classes,
                                                    ds_handle=test_ds,
                                                    freeze_base_model=True)

    checkpoint_path = "/Volumes/Data/tmp/orchids-models/orchid2019/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.RMSprop(),
                  metrics=['accuracy'])

    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)

    loss, accuracy = model.evaluate(test_ds)
    print('Test accuracy :', accuracy)
