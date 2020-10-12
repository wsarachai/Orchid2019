
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow import keras

import lib_utils


if __name__ == '__main__':
    print(tf.version.VERSION)

    train_ds, val_ds, test_ds, num_classes = lib_utils.create_dataset()
    model = lib_utils.create_orchid_mobilenet_v2_14(num_classes=num_classes,
                                                    freeze_base_model=True)

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

    #checkpoint_path = "/Volumes/Data/tmp/orchids-models/orchid2019/cp-{epoch:04d}.ckpt"
    checkpoint_path = "/Volumes/Data/tmp/orchids-models/orchid2019/cp-{epoch:04d}.h5"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    epochs = 0
    total_epochs = 100

    latest, step = lib_utils.latest_checkpoint(checkpoint_dir)
    if latest:
        epochs = step
        model.load_weights(latest, by_name=True)
    else:
        # Save the weights using the `checkpoint_path` format
        model.save_weights(checkpoint_path.format(epoch=0))

    # Create a callback that saves the model's weights
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=1)

    history_fine = model.fit(train_ds,
                             epochs=total_epochs,
                             validation_data=val_ds,
                             callbacks=[cp_callback],
                             initial_epoch=epochs)

    loss, accuracy = model.evaluate(test_ds, verbose=2)
    print('Test accuracy :', accuracy)