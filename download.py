"""
Script to help you download the dataset and pre-trained word2vec model
if you decide to train the model locally.
Otherwise, I would recommend you to train the model on the Kaggle notebook.
https://www.kaggle.com/blaisewang/topic-label-generation
"""

import os

import gdown
import wget


# create directory safely
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


directory = "./data"
create_directory(directory)

# download the topics
url = "https://raw.githubusercontent.com/sorodoc/multimodal_topic_label/master/data/topics.csv"
wget.download(url, directory)

# download the labels
url = "https://raw.githubusercontent.com/sorodoc/multimodal_topic_label/master/data/dataset_text.txt"
wget.download(url, directory)

# download the pre-trained word2vec model
directory = "./word2vec"
create_directory(directory)

url = "https://drive.google.com/uc?id=1-Rl6TluB6vCvs8oT7_0WPbZ2_nXBEyG8"
output = os.path.join(directory, "GoogleNews-vectors-negative300.bin")
gdown.download(url, output, quiet=False)
