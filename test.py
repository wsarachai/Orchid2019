import numpy as np
import tensorflow as tf


if __name__ == '__main__':
    var = tf.Variable(np.random.random(size=(1,)))

    def lr_callable():
        return .1

    opt = tf.keras.optimizers.SGD(learning_rate=lr_callable)
    loss = lambda: 3 * var
    opt.minimize(loss, var_list=[var])
