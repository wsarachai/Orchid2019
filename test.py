import os
import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    # learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=0.01, decay_steps=10, decay_rate=0.96
    # )
    learning_rate = 0.01

    optimizers = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizers,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    checkpoint_path = "/Users/watcharinsarachai/tmp/logs/training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     #save_weights_only=True,
                                                     verbose=1)

    def scheduler(epoch, lr):
        tf.summary.scalar("lr", lr, epoch)
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)


    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    w = tf.summary.create_file_writer('/Users/watcharinsarachai/tmp/logs/training_1')
    with w.as_default():
        model.fit(train_images, train_labels, epochs=10, callbacks=[callback, cp_callback])

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)

    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])

    predictions = probability_model.predict(test_images)

    np.argmax(predictions[0])