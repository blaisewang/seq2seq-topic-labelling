# Topic Label Generation with TensorFlow

Multiple encoder-decoder _seq-to-seq_ neural networks were trained for topic label generation task.

## 1. Word Embedding

### 1.1 Pre-Trained Word2Vec Model

[GoogleNews-vectors-negative300.bin.gz](https://code.google.com/archive/p/word2vec/) by [Mikolov et al. 2013](https://arxiv.org/abs/1310.4546).

### 1.2 tf.keras.layers.Embedding

Turns positive integers (indexes) into dense vectors of fixed size.

## 2. Neural Network Architecture

### 2.1 Recurrent Layers

#### 2.1.1 Simple RNN

Fully-connected RNN where the output is to be fed back to input.

#### 2.1.2 LSTM

Long Short-Term Memory layer - [Hochreiter 1997](https://www.mitpressjournals.org/doi/10.1162/neco.1997.9.8.1735).

#### 2.1.3 GRU

Gated Recurrent Unit - [Cho et al. 2014](https://arxiv.org/abs/1406.1078).

### 2.2 Encoder-Decoder

#### 2.2.1 Encoder

In the baseline model, the densely-connected layer is used to replace the recurrent layers.

Bidirectional RNN - [Schuster and Paliwa 1997](https://ieeexplore.ieee.org/document/650093) is compared with the unidirectional RNN.

#### 2.2.2 Decoder

**Output Shape:** (_BATCH_SIZE_, _VOCABULARY_SIZE_)

### 2.3 Attention Mechanism

[Bahdanau et al. 2014](https://arxiv.org/abs/1409.0473).

## 3. Training

- Teacher forcing - [Williams and Zipser 1989](https://www.mitpressjournals.org/doi/10.1162/neco.1989.1.2.270).

- Early stopping: _PATIENCE_ = 6.

- Learning rate scheduler used in [Vaswani et al. 2019 's paper](https://arxiv.org/abs/1706.03762).

## 4. Evaluation Metrics

- BLEU - [Papineni et al. 2002](https://www.aclweb.org/anthology/P02-1040/).

- GLEU (Google-BLEU) - [Wu et al. 2016](https://arxiv.org/abs/1609.08144).

- NIST - [Doddington 2002](https://dl.acm.org/citation.cfm?id=1289189.1289273).

- ROUGE - [Lin 2004](https://www.aclweb.org/anthology/W04-1013).

## 5. License

This code is distributed under the terms of the [BSD 3-Clause License](https://github.com/blaisewang/seq2seq-topic-labelling/blob/master/LICENSE).
