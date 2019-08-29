import tensorflow as tf


def rnn_layer(units):
    if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNGRU(units, return_sequences=True, return_state=True,
                                        recurrent_initializer="glorot_uniform")

    else:
        return tf.keras.layers.GRU(units, return_sequences=True, return_state=True,
                                   recurrent_activation="sigmoid", recurrent_initializer="glorot_uniform")
