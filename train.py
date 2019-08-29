"""
Script to help you train the model locally.
Otherwise, I would recommend you to train the model on the Kaggle notebook.
https://www.kaggle.com/blaisewang/topic-label-generation
"""

import csv
import math
import time

import gensim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tensorflow as tf
from nltk.translate import bleu_score, gleu_score, nist_score
from rouge import Rouge
from sklearn.model_selection import train_test_split

from models.pre_bi_gru_attn import Encoder, Decoder

# enable eager execution for TensorFlow < 2.0
tf.compat.v1.enable_eager_execution()

# data_15 means threshold value is 1.5
path_to_file = "./input/data_15.csv"

# word embedding
embedding_size = 300

# True for applying the early stopping
early_stopping = False

# True for splitting the same input sequences into different data sets
mix_input_topic = True

# True for applying the attention mechanism
decoder_attention = Decoder.attention_mechanism

# True for applying the pre-trained word2vec
pre_trained_word2vec = Encoder.pre_trained_word2vec

if pre_trained_word2vec and "model" not in locals():
    model = gensim.models.KeyedVectors.load_word2vec_format("./word2vec/GoogleNews-vectors-negative300.bin",
                                                            binary=True)
    vocab = model.vocab

    embedding_size = 300

    token_index = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}

    token_vector = {"<start>": tf.ones(embedding_size),
                    "<end>": tf.negative(tf.ones(embedding_size)),
                    "<unk>": tf.zeros(embedding_size),
                    "<pad>": tf.tile([0.5], [embedding_size])}


def preprocess_sentence(sent):
    if pre_trained_word2vec:
        return "<start> " + " ".join(topic if topic in vocab else "<unk>" for topic in sent.split()) + " <end>"
    return "<start> " + sent + " <end>"


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

            if pre_trained_word2vec:
                if all(label in token_vector for label in label_str.split()):
                    continue

            topics.append(topic_str)
            labels.append(label_str)

    return topics, labels


def max_length(vectors):
    return max(len(vector) for vector in vectors)


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="")
    lang_tokenizer.fit_on_texts(lang)

    indices_list = lang_tokenizer.texts_to_sequences(lang)

    indices_list = tf.keras.preprocessing.sequence.pad_sequences(indices_list, padding="post")

    return indices_list, lang_tokenizer


def create_reference_dict(inputs, targets):
    ref_dict = {}

    for topic, label in zip(inputs, targets):
        if topic not in ref_dict:
            ref_dict[topic] = []
        ref_dict[topic].append(label)

    return ref_dict


def input2vec(data):
    inputs = []
    targets = []

    for input_seq in data:
        for target_seq in reference_dict[input_seq]:
            inputs.append(input_seq)
            targets.append(target_seq)

    inputs = input_tokenizer.texts_to_sequences(inputs)
    targets = input_tokenizer.texts_to_sequences(targets)

    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=max_length_inp, padding="post")
    targets = tf.keras.preprocessing.sequence.pad_sequences(targets, maxlen=max_length_target, padding="post")

    return inputs, targets


def index2vec(index):
    if index <= 3:
        return token_vector[token_index[index]]
    return model.word_vec(model.index2word[index])


def indices2vec(indices):
    return [index2vec(int(index)) for index in indices]


# creating cleaned input, output pairs
input_lang, target_lang = create_dataset(path_to_file)

reference_dict = create_reference_dict(input_lang, target_lang)

input_vectors, input_tokenizer = tokenize(input_lang)
target_vectors, target_tokenizer = tokenize(target_lang)

# Calculate max_length of the vectors
max_length_inp, max_length_target = max_length(input_vectors), max_length(target_vectors)

# Creating training, val, test sets using an 70-20-10 split
if mix_input_topic:
    input_train, input_test, target_train, target_test = train_test_split(input_vectors, target_vectors, test_size=0.3)
    input_test, input_val, target_test, target_val = train_test_split(input_test, target_test, test_size=0.67)
else:
    input_train, input_test = train_test_split(list(reference_dict.keys()), test_size=0.3)
    input_val, input_test = train_test_split(input_test, test_size=0.67)

    input_train, target_train = input2vec(input_train)
    input_val, target_val = input2vec(input_val)
    input_test, target_test = input2vec(input_test)

BATCH_SIZE = 64
buffer_size = len(input_train)
vocab_inp_size = len(input_tokenizer.word_index) + 1
vocab_tar_size = len(target_tokenizer.word_index) + 1
train_steps_per_epoch = math.ceil(len(input_train) / BATCH_SIZE)
val_steps_per_epoch = math.ceil(len(input_val) / BATCH_SIZE)
test_steps_per_epoch = math.ceil(len(input_test) / BATCH_SIZE)

train_dataset = tf.data.Dataset.from_tensor_slices((input_train, target_train))
train_dataset = train_dataset.shuffle(buffer_size).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices((input_val, target_val))
val_dataset = val_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices((input_test, target_test))
test_dataset = test_dataset.batch(BATCH_SIZE)

# RNN units dimension
RNN_DIMENSION = 1024

# initialise encoder decoder with pre_trained_word2vec flag
if pre_trained_word2vec:
    encoder, decoder = Encoder(RNN_DIMENSION), Decoder(vocab_tar_size, RNN_DIMENSION)
else:
    encoder = Encoder(vocab_inp_size, embedding_size, RNN_DIMENSION)
    decoder = Decoder(vocab_tar_size, embedding_size, RNN_DIMENSION)


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
    enc_output = None

    with tf.GradientTape() as tape:

        if pre_trained_word2vec:
            inputs = [indices2vec(indices) for indices in inputs]

        if decoder_attention:
            enc_output, enc_hidden = encoder(inputs)
        else:
            enc_hidden = encoder(inputs)

        dec_hidden = enc_hidden

        if pre_trained_word2vec:
            dec_input = tf.expand_dims(tf.expand_dims(index2vec(target_tokenizer.word_index["<start>"]), 0), 0)
            dec_input = tf.tile(dec_input, [targets.shape[0], 1, 1])
        else:
            dec_input = tf.expand_dims([target_tokenizer.word_index["<start>"]] * targets.shape[0], 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targets.shape[1]):
            # passing enc_output to the decoder
            if decoder_attention:
                predictions, dec_hidden, _ = decoder(dec_input, state=dec_hidden, encoder_output=enc_output)
            else:
                predictions, dec_hidden = decoder(dec_input, state=dec_hidden)

            loss += loss_function(targets[:, t], predictions)
            train_accuracy.update_state(targets[:, t], predictions)

            # using teacher forcing
            if pre_trained_word2vec:
                dec_input = tf.expand_dims(indices2vec(targets[:, t]), 1)
            else:
                dec_input = tf.expand_dims(targets[:, t], 1)

    train_loss((loss / int(targets.shape[1])))

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))


def test_step(inputs, targets):
    loss = 0
    enc_output = None

    if pre_trained_word2vec:
        inputs = [indices2vec(indices) for indices in inputs]

    if decoder_attention:
        enc_output, enc_hidden = encoder(inputs)
    else:
        enc_hidden = encoder(inputs)

    dec_hidden = enc_hidden

    predicted_labels = []

    if pre_trained_word2vec:
        dec_input = tf.expand_dims(tf.expand_dims(index2vec(target_tokenizer.word_index["<start>"]), 0), 0)
        dec_input = tf.tile(dec_input, [targets.shape[0], 1, 1])
    else:
        dec_input = tf.expand_dims([target_tokenizer.word_index["<start>"]] * targets.shape[0], 1)

    for t in range(1, max_length_target):
        # passing enc_output to the decoder
        if decoder_attention:
            predictions, dec_hidden, _ = decoder(dec_input, state=dec_hidden, encoder_output=enc_output)
        else:
            predictions, dec_hidden = decoder(dec_input, state=dec_hidden)

        loss += loss_function(targets[:, t], predictions)
        test_accuracy.update_state(targets[:, t], predictions)

        predicted = tf.math.argmax(predictions, axis=1)
        predicted_labels.append(list(predicted.numpy()))

        if pre_trained_word2vec:
            dec_input = tf.expand_dims(indices2vec(predicted), 1)
        else:
            dec_input = tf.expand_dims(predicted, 1)

    test_loss((loss / int(targets.shape[1])))

    result = []
    for label in zip(*predicted_labels):
        result.append([])
        for value in label:
            if value in (0, 2):
                break
            result[-1].append(value)

    return result


def word_split(sent):
    return [label.split()[1:-1] for label in reference_dict[sent]]


def rouge_sum_score(rouge_dict):
    return sum(value for fpr in rouge_dict.values() for value in fpr.values())


def rouge_dict_format(rouge_dict):
    return "{rouge-1: {f: %f, p: %f, r: %f}, rouge-l: {f: %f, p: %f, r: %f}}" % (
        rouge_dict["rouge-1"]["f"], rouge_dict["rouge-1"]["p"], rouge_dict["rouge-1"]["r"],
        rouge_dict["rouge-l"]["f"], rouge_dict["rouge-l"]["p"], rouge_dict["rouge-l"]["r"])


def evaluation_metrics(dataset, steps, size):
    references = []
    hypotheses = []

    rouge = Rouge()
    rouge_dict = {"rouge-1": {"f": 0.0, "p": 0.0, "r": 0.0},
                  "rouge-2": {"f": 0.0, "p": 0.0, "r": 0.0},
                  "rouge-l": {"f": 0.0, "p": 0.0, "r": 0.0}}

    for inputs, targets in dataset.take(steps):
        for labels in target_tokenizer.sequences_to_texts(test_step(inputs, targets)):
            if len(labels) > 0:
                hypotheses.append(labels.split())
            else:
                hypotheses.append([""])

        for labels in input_tokenizer.sequences_to_texts(inputs.numpy()):
            references.append(word_split(labels))

    for index, hypothesis in enumerate(hypotheses):
        max_score = {"rouge-1": {"f": 0.0, "p": 0.0, "r": 0.0},
                     "rouge-2": {"f": 0.0, "p": 0.0, "r": 0.0},
                     "rouge-l": {"f": 0.0, "p": 0.0, "r": 0.0}}

        for reference in references[index]:
            try:
                score = rouge.get_scores(" ".join(hypothesis), " ".join(reference))[0]
                if rouge_sum_score(score) > rouge_sum_score(max_score):
                    max_score = score
            except ValueError:
                pass

        for method_key in rouge_dict:
            for fpr in rouge_dict[method_key]:
                rouge_dict[method_key][fpr] += max_score[method_key][fpr]

    for method_key in rouge_dict:
        for fpr in rouge_dict[method_key]:
            rouge_dict[method_key][fpr] /= size

    print("BLEU-1 Score: %.4f" % bleu_score.corpus_bleu(references, hypotheses, weights=(1,)))
    print("GLEU-1 Score: %.4f" % gleu_score.corpus_gleu(references, hypotheses, max_len=1))
    print("NIST-1 Score: %.4f" % nist_score.corpus_nist(references, hypotheses, n=1))
    print("ROUGE Scores: %s" % rouge_dict_format(rouge_dict))


EPOCH = 20
PATIENCE = 5
stop_flags = []
last_loss = float("inf")

for epoch in range(EPOCH):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    test_loss.reset_states()
    test_accuracy.reset_states()

    train_dataset = train_dataset.shuffle(buffer_size)

    for inp, target in train_dataset.take(train_steps_per_epoch):
        train_step(inp, target)

    evaluation_metrics(val_dataset, val_steps_per_epoch, len(input_val))

    print("Train Loss: %.4f Accuracy: %.4f" % (train_loss.result(), train_accuracy.result()))
    print("Validation Loss: %.4f Accuracy: %.4f" % (test_loss.result(), test_accuracy.result()))
    print("%.4f secs taken for epoch %d\n" % (time.time() - start, epoch + 1))

    # early stopping
    if early_stopping:
        if test_loss.result() > last_loss or abs(last_loss - test_loss.result()) < 1e-4:
            stop_flags.append(True)
        else:
            stop_flags.clear()

        if len(stop_flags) >= PATIENCE:
            print("\nEarly stopping\n")
            break

        last_loss = test_loss.result()

test_loss.reset_states()
test_accuracy.reset_states()

evaluation_metrics(test_dataset, test_steps_per_epoch, len(input_test))

print("Test Loss: %.4f Accuracy: %.4f\n" % (test_loss.result(), test_accuracy.result()))


# function for plotting the attention weights
def plot_attention(attention, result, sentence):
    result = result.split()
    sentence = sentence.split()
    attention = attention[:len(result), :len(sentence)]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap="viridis")

    font_dict = {"fontsize": 14}

    ax.set_xticklabels([""] + sentence, fontdict=font_dict, rotation=90)
    ax.set_yticklabels([""] + result, fontdict=font_dict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def sample_evaluation(sentence):
    result = ""
    enc_out = None
    sentence = preprocess_sentence(sentence)
    attention_plot = np.zeros((max_length_target, max_length_inp))

    inputs = [input_tokenizer.word_index[w] for w in sentence.split(" ")]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding="post")

    if pre_trained_word2vec:
        inputs = [indices2vec(indices) for indices in inputs]
    else:
        inputs = tf.convert_to_tensor(inputs)

    if decoder_attention:
        enc_out, enc_hidden = encoder(inputs)
    else:
        enc_hidden = encoder(inputs)

    dec_hidden = enc_hidden

    if pre_trained_word2vec:
        dec_input = tf.expand_dims(tf.expand_dims(index2vec(target_tokenizer.word_index["<start>"]), 0), 0)
    else:
        dec_input = tf.expand_dims([target_tokenizer.word_index["<start>"]], 0)

    for t in range(max_length_target):

        if decoder_attention:
            predictions, dec_hidden, attention_weights = decoder(dec_input, state=dec_hidden, encoder_output=enc_out)

            # storing the attention weights to plot later on
            attention_weights = tf.reshape(attention_weights, (-1,))
            attention_plot[t] = attention_weights.numpy()
        else:
            predictions, dec_hidden = decoder(dec_input, state=dec_hidden)

        predicted_id = tf.math.argmax(predictions[0]).numpy()
        result += target_tokenizer.index_word[predicted_id] + " "

        if target_tokenizer.index_word[predicted_id] == "<end>":
            break

        # the predicted ID is fed back into the model
        if pre_trained_word2vec:
            dec_input = tf.expand_dims([index2vec(predicted_id)], 1)
        else:
            dec_input = tf.expand_dims([predicted_id], 0)

    if decoder_attention:
        result = result.strip()
        plot_attention(attention_plot, result, sentence)

    return result, sentence


def generate_topic(sentence):
    result, sentence = sample_evaluation(sentence)

    print("Input labels: %s" % sentence)
    print("Predicted topic: %s" % "<start> " + result)
    if sentence in reference_dict:
        print("Target topic: %s\n" % ', '.join(reference_dict[sentence]))


generate_topic("system cost datum tool analysis provide design technology develop information")

generate_topic("treatment patient trial therapy study month week efficacy effect receive")

generate_topic("case report lesion present rare diagnosis lymphoma mass cyst reveal")

generate_topic("film movie star director hollywood actor minute direct story witch")

generate_topic("cup cook minute add pepper salt serve tablespoon oil sauce")
