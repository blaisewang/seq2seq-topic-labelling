import csv
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

tf.compat.v1.enable_eager_execution()

path_to_file = "../input/data.csv"


def preprocess_sentence(sent):
    return "<start> " + sent + " <end>"


# Return topic label pairs
def create_dataset(path):
    topics = []
    labels = []

    with open(path, "r") as csv_data:
        reader = csv.reader(csv_data)

        # skip header
        next(reader, None)

        for row in reader:
            topics.append(preprocess_sentence(row[0]))
            labels.append(preprocess_sentence(row[1]))

    return topics, labels


def max_length(tensor):
    return max(len(num) for num in tensor)


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="")
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding="post")

    return tensor, lang_tokenizer


# creating cleaned input, output pairs
input_lang, target_lang = create_dataset(path_to_file)

input_tensor, input_lang_tokenizer = tokenize(input_lang)
target_tensor, target_lang_tokenizer = tokenize(target_lang)

# Calculate max_length of the target tensors
max_length_inp, max_length_target = max_length(input_tensor), max_length(target_tensor)

# Creating training and test sets using an 70-30 split
input_train, input_test, target_train, target_test = train_test_split(input_tensor, target_tensor, test_size=0.3)

# Creating test and validation sets using 15-15 split, 70-15-15
input_test, input_val, target_test, target_val = train_test_split(input_test, target_test, test_size=0.5)

BUFFER_SIZE = len(input_train)
BATCH_SIZE = 64
train_steps_per_epoch = len(input_train) // BATCH_SIZE
val_steps_per_epoch = len(input_val) // BATCH_SIZE
test_steps_per_epoch = len(input_test) // BATCH_SIZE
embedding_dimension = 256
dimensionality = 1024
vocab_inp_size = len(input_lang_tokenizer.word_index) + 1
vocab_tar_size = len(target_lang_tokenizer.word_index) + 1

train_dataset = tf.data.Dataset.from_tensor_slices((input_train, target_train)).shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)

val_dataset = tf.data.Dataset.from_tensor_slices((input_val, target_val))
val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)

test_dataset = tf.data.Dataset.from_tensor_slices((input_test, target_test))
test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True)


def lstm(units):
    if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNLSTM(units, return_sequences=True, return_state=True,
                                         recurrent_initializer="glorot_uniform")

    else:
        return tf.keras.layers.LSTM(units, return_sequences=True, return_state=True,
                                    recurrent_activation="sigmoid", recurrent_initializer="glorot_uniform")


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.rnn = tf.keras.layers.Bidirectional(lstm(self.enc_units), merge_mode="concat")

    def call(self, x):
        x = self.embedding(x)
        output, forward_h, forward_c, backward_h, backward_c = self.rnn(x)
        state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
        state_c = tf.keras.layers.Concatenate()([forward_c, backward_c])
        state = tf.add(state_h, state_c)
        return output, state


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = lstm(self.dec_units * 2)
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, encoder_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, encoder_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, h, c = self.lstm(x)

        state = tf.add(h, c)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, warm_up_steps=2000):
        super(CustomSchedule, self).__init__()

        self.warm_up_steps = warm_up_steps

        self.d_model = int(4000 / warm_up_steps) * 128
        self.d_model = tf.cast(self.d_model, tf.float32)

    def get_config(self):
        pass

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warm_up_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


encoder = Encoder(vocab_inp_size, embedding_dimension, dimensionality)
decoder = Decoder(vocab_tar_size, embedding_dimension, dimensionality)

learning_rate = CustomSchedule()

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


train_loss = tf.keras.metrics.Mean(name="train_loss")
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

test_loss = tf.keras.metrics.Mean(name="test_loss")
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")


def train_step(inputs, targets):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inputs)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([target_lang_tokenizer.word_index["<start>"]] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targets.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targets[:, t], predictions)
            train_accuracy.update_state(targets[:, t], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(targets[:, t], 1)

    train_loss((loss / int(targets.shape[1])))

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))


def test_step(inputs, targets):
    loss = 0

    enc_output, enc_hidden = encoder(inputs)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([target_lang_tokenizer.word_index["<start>"]] * BATCH_SIZE, 1)

    for t in range(1, max_length_target):
        # passing enc_output to the decoder
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

        loss += loss_function(targets[:, t], predictions)
        test_accuracy.update_state(targets[:, t], predictions)

        predicted = tf.math.argmax(predictions, axis=1)

        # using teacher forcing
        dec_input = tf.expand_dims(predicted, 1)

    test_loss((loss / int(targets.shape[1])))


EPOCHS = 10
PATIENCE = 5

stop_flags = []
last_val_accuracy = 0

for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    test_loss.reset_states()
    test_accuracy.reset_states()

    train_dataset = train_dataset.shuffle(BUFFER_SIZE)

    for inp, target in train_dataset.take(train_steps_per_epoch):
        train_step(inp, target)

    for inp, target in val_dataset.take(val_steps_per_epoch):
        test_step(inp, target)

    print("Train Loss: %.4f Accuracy: %.4f" % (train_loss.result(), train_accuracy.result()))
    print("Validation Loss: %.4f Accuracy: %.4f" % (test_loss.result(), test_accuracy.result()))
    print("%.4f secs taken for epoch %d\n" % (time.time() - start, epoch + 1))

    if test_accuracy.result() < last_val_accuracy or abs(last_val_accuracy - test_accuracy.result()) < 1e-4:
        stop_flags.append(True)
    else:
        stop_flags.clear()

    # if len(stop_flags) >= PATIENCE:
    #     print("\nEarly stopping\n")
    #     break

    last_val_accuracy = test_accuracy.result()

test_loss.reset_states()
test_accuracy.reset_states()

for inp, target in test_dataset.take(test_steps_per_epoch):
    test_step(inp, target)

print("Test Loss: %.4f Accuracy: %.4f\n" % (test_loss.result(), test_accuracy.result()))


def evaluate(sentence):
    attention_plot = np.zeros((max_length_target, max_length_inp))

    sentence = preprocess_sentence(sentence)

    inputs = [input_lang_tokenizer.word_index[w] for w in sentence.split(" ")]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding="post")
    inputs = tf.convert_to_tensor(inputs)

    result = ""

    enc_out, enc_hidden = encoder(inputs)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([target_lang_tokenizer.word_index["<start>"]], 0)

    for t in range(max_length_target):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1,))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.math.argmax(predictions[0]).numpy()

        result += target_lang_tokenizer.index_word[predicted_id] + " "

        if target_lang_tokenizer.index_word[predicted_id] == "<end>":
            return result, sentence, attention_plot

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot


# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap="viridis")

    font_dict = {"fontsize": 14}

    ax.set_xticklabels([""] + sentence, fontdict=font_dict, rotation=90)
    ax.set_yticklabels([""] + predicted_sentence, fontdict=font_dict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def generate_topic(sentence):
    result, sentence, attention_plot = evaluate(sentence)

    print("Input labels: %s" % sentence)
    print("Predicted topic: %s" % result)

    # attention_plot = attention_plot[:len(result.split(" ")), :len(sentence.split(" "))]
    # plot_attention(attention_plot, sentence.split(" "), result.split(" "))


generate_topic("system cost datum tool analysis provide design technology develop information")

generate_topic("treatment patient trial therapy study month week efficacy effect receive")

generate_topic("case report lesion present rare diagnosis lymphoma mass cyst reveal")

generate_topic("film movie star director hollywood actor minute direct story witch")
