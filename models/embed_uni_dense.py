"""
Unidirectional densely-connected NN encoder and GRU decoder with word embedding
"""

import tensorflow as tf

from layers.gru import rnn_layer


class Encoder(tf.keras.Model):
    pre_trained_word2vec = False

    def __init__(self, vocab_size, embedding_size, enc_units):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(enc_units, activation='tanh')

    def call(self, x, **kwargs):
        x = tf.cast(x, dtype=tf.float32)

        x = self.embedding(x)

        # flatten the x
        x = self.flatten(x)
        output = self.dense(x)

        return output


class Decoder(tf.keras.Model):
    attention_mechanism = False

    def __init__(self, vocab_size, embedding_size, dec_units):
        super(Decoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.rnn = rnn_layer(dec_units)
        self.fc = tf.keras.layers.Dense(vocab_size, activation="softmax")

    def call(self, x, **kwargs):
        # x shape == (batch_size, 1, embedding_dim)
        x = tf.cast(x, dtype=tf.float32)

        state = kwargs["state"]

        # x shape == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(state, 1), x], axis=-1)

        # passing the concatenated vector to the rnn layer
        output, state = self.rnn(x, initial_state=state)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state
