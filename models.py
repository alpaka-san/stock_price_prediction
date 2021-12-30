import tensorflow as tf


def Model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(400, input_shape = (5, 1), activation = tf.nn.leaky_relu, return_sequences = True),
        tf.keras.layers.LSTM(400, activation = tf.nn.leaky_relu),
        tf.keras.layers.Dense(400, activation = tf.nn.leaky_relu),
        tf.keras.layers.Dense(200, activation = tf.nn.leaky_relu),
        tf.keras.layers.Dense(100, activation = tf.nn.leaky_relu),
        tf.keras.layers.Dense(5, activation = tf.nn.leaky_relu)
    ])
    return model