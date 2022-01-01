import tensorflow as tf


def Model():
    model = tf.keras.models.Sequential([
        # tf.keras.layers.LSTM(800, input_shape = (10, 1), activation = tf.nn.leaky_relu, return_sequences = True),
        # tf.keras.layers.LSTM(800, activation = tf.nn.leaky_relu),
        # tf.keras.layers.Dense(400, activation = tf.nn.leaky_relu),
        # tf.keras.layers.Dense(200, activation = tf.nn.leaky_relu),
        # tf.keras.layers.Dense(100, activation = tf.nn.leaky_relu),
        # tf.keras.layers.Dense(5, activation = tf.nn.leaky_relu)
        tf.keras.layers.LSTM(50, input_shape = (10, 1), activation = tf.nn.leaky_relu, return_sequences = True),
        tf.keras.layers.LSTM(50, activation = tf.nn.leaky_relu),
        tf.keras.layers.Dense(25, activation = tf.nn.leaky_relu),
        tf.keras.layers.Dense(5, activation = tf.nn.leaky_relu),
    ])
    return model