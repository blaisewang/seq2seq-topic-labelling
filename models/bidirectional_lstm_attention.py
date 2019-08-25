import csv
import math
import time

import gensim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tensorflow as tf
from nltk.translate import bleu_score
from nltk.translate import gleu_score
from nltk.translate import nist_score
from sklearn.model_selection import train_test_split

tf.compat.v1.enable_eager_execution()

path_to_file = "./input/data.csv"

# load pre-trained word2vec model
model = gensim.models.KeyedVectors.load_word2vec_format("./word2vec/GoogleNews-vectors-negative300.bin", binary=True)

vocab = model.vocab

EMBEDDING_SIZE = 300

SYMBOL_INDEX = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}

SYMBOL_VALUE = {"<start>": tf.ones(EMBEDDING_SIZE),
                "<end>": tf.negative(tf.ones(EMBEDDING_SIZE)),
                "<unk>": tf.zeros(EMBEDDING_SIZE),
                "<pad>": tf.tile([0.5], [300])}


def preprocess_sentence(sent):
    return "<start> " + " ".join(topic if topic in vocab else "<unk>" for topic in sent.split()) + " <end>"


# Return topic label pairs
def create_dataset(path):
    topics = []
    labels = []

    with open(path, "r") as csv_data:
        reader = csv.reader(csv_data)

        # skip header
        next(reader, None)

        for row in reader:
            topic_str = preprocess_sentence(row[0])
            label_str = preprocess_sentence(row[1])

            if all(label in SYMBOL_VALUE for label in label_str.split()):
                continue

            topics.append(topic_str)
            labels.append(label_str)

    return topics, labels


def index2vec(index):
    if index <= 3:
        return SYMBOL_VALUE[SYMBOL_INDEX[index]]
    return model.word_vec(model.index2word[index])


def indices2vec(indices):
    return [index2vec(int(index)) for index in indices]


def max_length(vectors):
    return max(len(vector) for vector in vectors)


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="")
    lang_tokenizer.fit_on_texts(lang)

    indices_list = lang_tokenizer.texts_to_sequences(lang)

    indices_list = tf.keras.preprocessing.sequence.pad_sequences(indices_list, padding="post")

    return indices_list, lang_tokenizer


def reference_dict(inputs, targets):
    ref_dict = {}

    for topic, label in zip(inputs, targets):
        if topic not in ref_dict:
            ref_dict[topic] = []
        ref_dict[topic].append(label)

    return ref_dict


# creating cleaned input, output pairs
input_lang, target_lang = create_dataset(path_to_file)

references = reference_dict(input_lang, target_lang)

input_vectors, input_tokenizer = tokenize(input_lang)
target_vectors, target_tokenizer = tokenize(target_lang)

# assert
assert all(input_tokenizer.index_word[i] == SYMBOL_INDEX[i] for i in [1, 2, 3])
assert all(target_tokenizer.index_word[i] == SYMBOL_INDEX[i] for i in [1, 2, 3])

# Calculate max_length of the vectors
max_length_inp, max_length_target = max_length(input_vectors), max_length(target_vectors)

# Creating training and test sets using an 70-30 split
input_train, input_test, target_train, target_test = train_test_split(input_vectors, target_vectors, test_size=0.3)

# Creating test and validation sets using 15-15 split, 70-15-15
input_test, input_val, target_test, target_val = train_test_split(input_test, target_test, test_size=0.5)

BUFFER_SIZE = len(input_train)
BATCH_SIZE = 64
train_steps_per_epoch = math.ceil(len(input_train) / BATCH_SIZE)
val_steps_per_epoch = math.ceil(len(input_val) / BATCH_SIZE)
test_steps_per_epoch = math.ceil(len(input_test) / BATCH_SIZE)
dimensionality = 1024
vocab_inp_size = len(input_tokenizer.word_index) + 1
vocab_tar_size = len(target_tokenizer.word_index) + 1

train_dataset = tf.data.Dataset.from_tensor_slices((input_train, target_train))
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices((input_val, target_val))
val_dataset = val_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices((input_test, target_test))
test_dataset = test_dataset.batch(BATCH_SIZE)


def lstm(units):
    if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNLSTM(units, return_sequences=True, return_state=True,
                                         recurrent_initializer="glorot_uniform")

    else:
        return tf.keras.layers.LSTM(units, return_sequences=True, return_state=True,
                                    recurrent_activation="sigmoid", recurrent_initializer="glorot_uniform")


class Encoder(tf.keras.Model):
    def __init__(self, enc_units):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.rnn = tf.keras.layers.Bidirectional(lstm(self.enc_units), merge_mode="concat")

    def call(self, x):
        x = tf.cast(x, dtype=tf.float32)

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
    def __init__(self, vocab_size, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.lstm = lstm(self.dec_units * 2)
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, encoder_output):
        # x shape == (batch_size, 1, embedding_dim)
        x = tf.cast(x, dtype=tf.float32)

        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, encoder_output)

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


encoder = Encoder(dimensionality)
decoder = Decoder(vocab_tar_size, dimensionality)

learning_rate = CustomSchedule()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
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
        # vectors = [indices2vec(indices) for indices in indices_list]

        inp_vectors = [indices2vec(indices) for indices in inputs]

        enc_output, enc_hidden = encoder(inp_vectors)
        dec_hidden = enc_hidden

        # dec_input = tf.expand_dims([target_lang_tokenizer.word_index["<start>"]] * BATCH_SIZE, 1)
        dec_input = tf.expand_dims(tf.expand_dims(index2vec(target_tokenizer.word_index["<start>"]), 0), 0)
        dec_input = tf.tile(dec_input, [inputs.shape[0], 1, 1])

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targets.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targets[:, t], predictions)
            train_accuracy.update_state(targets[:, t], predictions)

            # using teacher forcing
            # dec_input = tf.expand_dims(targets[:, t], 1)
            dec_input = tf.expand_dims(indices2vec(targets[:, t]), 1)

    train_loss((loss / int(targets.shape[1])))

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))


def test_step(inputs, targets):
    loss = 0

    inp_vectors = [indices2vec(indices) for indices in inputs]

    enc_output, enc_hidden = encoder(inp_vectors)
    dec_hidden = enc_hidden

    predicted_labels = []

    # dec_input = tf.expand_dims([target_lang_tokenizer.word_index["<start>"]] * BATCH_SIZE, 1)
    dec_input = tf.expand_dims(tf.expand_dims(index2vec(target_tokenizer.word_index["<start>"]), 0), 0)
    dec_input = tf.tile(dec_input, [inputs.shape[0], 1, 1])

    for t in range(1, max_length_target):
        # passing enc_output to the decoder
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

        loss += loss_function(targets[:, t], predictions)
        test_accuracy.update_state(targets[:, t], predictions)

        predicted = tf.math.argmax(predictions, axis=1)
        predicted_labels.append(list(predicted.numpy()))

        # dec_input = tf.expand_dims(predicted, 1)
        dec_input = tf.expand_dims(indices2vec(predicted), 1)

    test_loss((loss / int(targets.shape[1])))

    result = []
    for label in zip(*predicted_labels):
        result.append([])
        for value in label:
            if value in (0, 2):
                break
            result[-1].append(value)

    return result


def evaluation(dataset, steps):
    eval_references = []
    eval_hypotheses = []

    for inputs, targets in dataset.take(steps):
        for labels in target_tokenizer.sequences_to_texts(test_step(inputs, targets)):
            if len(labels) > 0:
                eval_hypotheses.append(labels.split())
            else:
                eval_hypotheses.append([""])

        for labels in input_tokenizer.sequences_to_texts(inputs.numpy()):
            eval_references.append(word_split(labels))

    print("BLUE-1 Score: %f" % bleu_score.corpus_bleu(eval_references, eval_hypotheses, weights=(1,)))
    print("GLUE-1 Score: %f" % gleu_score.corpus_gleu(eval_references, eval_hypotheses, max_len=1))
    print("NIST-1 Score: %f" % nist_score.corpus_nist(eval_references, eval_hypotheses, n=1))


def word_split(sent):
    return [label.split()[1:-1] for label in references[sent]]


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

    evaluation(val_dataset, val_steps_per_epoch)

    print("Train Loss: %.4f Accuracy: %.4f" % (train_loss.result(), train_accuracy.result()))
    print("Validation Loss: %.4f Accuracy: %.4f" % (test_loss.result(), test_accuracy.result()))
    print("%.4f secs taken for epoch %d\n" % (time.time() - start, epoch + 1))

    # if test_accuracy.result() < last_val_accuracy or abs(last_val_accuracy - test_accuracy.result()) < 1e-4:
    #     stop_flags.append(True)
    # else:
    #     stop_flags.clear()
    #
    # if len(stop_flags) >= PATIENCE:
    #     print("\nEarly stopping\n")
    #     break
    #
    # last_val_accuracy = test_accuracy.result()

test_loss.reset_states()
test_accuracy.reset_states()

evaluation(test_dataset, test_steps_per_epoch)

print("Test Loss: %.4f Accuracy: %.4f\n" % (test_loss.result(), test_accuracy.result()))


def evaluate(sentence):
    attention_plot = np.zeros((max_length_target, max_length_inp))

    sentence = preprocess_sentence(sentence)

    inputs = [input_tokenizer.word_index[w] for w in sentence.split(" ")]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding="post")
    inputs = [indices2vec(indices) for indices in inputs]

    result = ""

    enc_out, enc_hidden = encoder(inputs)

    dec_hidden = enc_hidden

    # dec_input = tf.expand_dims([target_lang_tokenizer.word_index["<start>"]], 0)
    dec_input = tf.expand_dims(tf.expand_dims(index2vec(target_tokenizer.word_index["<start>"]), 0), 0)

    for t in range(max_length_target):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1,))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.math.argmax(predictions[0]).numpy()

        result += target_tokenizer.index_word[predicted_id] + " "

        if target_tokenizer.index_word[predicted_id] == "<end>":
            return result.strip(), sentence, attention_plot

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([index2vec(predicted_id)], 1)

    return result.strip(), sentence, attention_plot


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
    print("Predicted topic: %s" % "<start> " + result)
    if sentence in references:
        print("Target topic: %s\n" % ', '.join(references[sentence]))

    # attention_plot = attention_plot[:len(result.split(" ")), :len(sentence.split(" "))]
    # plot_attention(attention_plot, sentence.split(" "), result.split(" "))


generate_topic("system cost datum tool analysis provide design technology develop information")

generate_topic("treatment patient trial therapy study month week efficacy effect receive")

generate_topic("case report lesion present rare diagnosis lymphoma mass cyst reveal")

generate_topic("film movie star director hollywood actor minute direct story witch")
