import tensorflow as tf


def Model():
    """
    Model definition. 
    
    Parameters:
        input (N, 5, 1): N is the number of sequential data. 
        output (N, 5, 1):
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(50, input_shape = (5, 1), activation = tf.nn.leaky_relu, return_sequences = True),
        tf.keras.layers.LSTM(50, activation = tf.nn.leaky_relu),
        tf.keras.layers.Dense(25, activation = tf.nn.leaky_relu),
        tf.keras.layers.Dense(5, activation = tf.nn.leaky_relu),
    ])
    return model