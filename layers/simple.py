import tensorflow as tf


def rnn_layer(units):
    return tf.keras.layers.SimpleRNN(units, return_sequences=True, return_state=True,
                                     recurrent_initializer="glorot_uniform")
