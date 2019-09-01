import tensorflow as tf

l2 = tf.keras.regularizers.l2(0.02)


def rnn_layer(units):
    return tf.keras.layers.LSTM(units, recurrent_activation="sigmoid", recurrent_initializer="glorot_uniform",
                                kernel_regularizer=l2, recurrent_regularizer=l2, dropout=0.5, recurrent_dropout=0.5,
                                return_sequences=True, return_state=True)
