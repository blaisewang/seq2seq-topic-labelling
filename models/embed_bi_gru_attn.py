"""
Bidirectional GRU with attention and word embedding
"""

import tensorflow as tf

from layers.gru import rnn_layer
from .attention import BahdanauAttention


class Encoder(tf.keras.Model):
    pre_trained_word2vec = False

    def __init__(self, vocab_size, embedding_size, enc_units):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.rnn = tf.keras.layers.Bidirectional(rnn_layer(enc_units))

    def call(self, x, **kwargs):
        x = tf.cast(x, dtype=tf.float32)

        x = self.embedding(x)

        output, forward, backward, = self.rnn(x)

        # concatenate forward and backward
        state = tf.keras.layers.Concatenate()([forward, backward])

        return output, state


class Decoder(tf.keras.Model):
    attention_mechanism = True

    def __init__(self, vocab_size, embedding_size, dec_units):
        super(Decoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.rnn = rnn_layer(dec_units * 2)
        self.fc = tf.keras.layers.Dense(vocab_size, activation="softmax")

        # attention mechanism
        self.attention = BahdanauAttention(dec_units)

    def call(self, x, **kwargs):
        x = tf.cast(x, dtype=tf.float32)

        state = kwargs["state"]
        encoder_output = kwargs["encoder_output"]

        # x shape == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(state, values=encoder_output)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the rnn layer
        output, state = self.rnn(x, initial_state=state)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights
