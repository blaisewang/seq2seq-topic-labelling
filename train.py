import csv

import numpy as np
from gensim.models import Word2Vec
from keras.layers import Dense, LSTM, Input, Activation, Add, TimeDistributed, Flatten, Multiply
from keras.models import Model
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

topics = []
labels = []

# dataset loading
with open("data/topics_cleaned.csv", "r") as topics_in, open("data/labels_cleaned.csv", "r") as labels_in:
    topic_reader = csv.reader(topics_in)
    label_reader = csv.reader(labels_in)

    for row in topic_reader:
        topics.append(row)
    for row in label_reader:
        labels.append(row[0].split())

# one-hot encoding
vocabulary = []

# for topic in topics:
#     for term in topic:
#         vocabulary.append(term)

for label in labels:
    for term in label:
        vocabulary.append(term)

vocabulary = list(sorted(set(vocabulary)))

one_hot_encoder = OneHotEncoder(sparse=False)
one_hot_encoded = one_hot_encoder.fit_transform(np.array(vocabulary).reshape(len(vocabulary), 1))

one_hot = {}

for index, word in enumerate(vocabulary):
    one_hot[word] = one_hot_encoded[index]

# word2vec model training

size = 100
corpus = topics + labels

word2vec_model = Word2Vec(corpus, size=size, compute_loss=True, window=10, min_count=1, workers=12, batch_words=20)
word2vec_model.train(corpus, total_examples=len(corpus), epochs=5)

# word2vec
x = []
y = []

for topic, label in zip(topics, labels):
    x_word = []
    for word in topic:
        x_word.append(word2vec_model.wv.word_vec(word))

    y_word = []
    for word in label:
        y_word.append(one_hot[word])

    x.append(x_word)
    y.append(y_word)

x = np.array(x)
y = np.array(y)

# zero padding
max_label_len = max(len(y_word) for y_word in y)
zero_padding = np.zeros(len(vocabulary))

for index, y_word in enumerate(y):
    diff = max_label_len - len(y_word)
    for _ in range(diff):
        y[index] = np.vstack([y[index], zero_padding])


# start sequence

def add_zeros(seq):
    return np.insert(seq, [0], [[0], ], axis=0)


y = np.array(list(map(add_zeros, y)))

# RNN structure

epochs = 20
batch_size = 50
learning_rate = 0.01

encoder_shape = np.shape(x[0])
decoder_shape = np.shape(y[0])

print(encoder_shape, decoder_shape)

"""__encoder___"""
encoder_inputs = Input(shape=encoder_shape)

encoder_LSTM_forward = LSTM(10, dropout=0.2, return_state=True, return_sequences=True)
encoder_LSTM_backward = LSTM(10, return_state=True, return_sequences=True, dropout=0.05, go_backwards=True)

encoder_outputs_f, state_h_f, state_c_f = encoder_LSTM_forward(encoder_inputs)
encoder_outputs_b, state_h_b, state_c_b = encoder_LSTM_backward(encoder_inputs)

state_h = Add()([state_h_f, state_h_b])
state_c = Add()([state_c_f, state_c_b])
encoder_outputs_final = Add()([encoder_outputs_f, encoder_outputs_b])

encoder_states = [state_h, state_c]

"""____decoder___"""
decoder_inputs = Input(shape=(None, decoder_shape[1]))
decoder_LSTM = LSTM(10, return_sequences=True, dropout=0.2, return_state=True)
decoder_outputs, _, _ = decoder_LSTM(decoder_inputs, initial_state=encoder_states)

attention = TimeDistributed(Dense(1, activation="tanh"))(encoder_outputs_final)
attention = Flatten()(attention)
attention = Multiply()([decoder_outputs, attention])
attention = Activation("softmax")(attention)
# attention = Permute([2, 1])(attention)

decoder_dense = Dense(decoder_shape[1], activation="softmax")
decoder_outputs = decoder_dense(attention)

model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)

print(model.summary())

# RNN training

rms_prop = RMSprop(lr=learning_rate)
model.compile(loss="categorical_crossentropy", optimizer=rms_prop, metrics=["accuracy"])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10)
history = model.fit(x=[x_train, y_train], y=y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                    validation_data=([x_test, y_test], y_test))

scores = model.evaluate([x_test, y_test], y_test, verbose=1)

print(scores)
