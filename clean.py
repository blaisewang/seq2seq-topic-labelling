"""
Script to help you clean the dataset if you decide to train the model locally.
Otherwise, I would recommend you to train the model on the Kaggle notebook.
https://www.kaggle.com/blaisewang/topic-label-generation
"""

import csv
import os
import re

DUMMY_LINE = "228.0	#	-1.0"


# create directory safely
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def clean_word(words):
    string = ""

    for word in words:

        word = word.replace(".", "")

        cleaned = re.findall(r"[\w/-]+", word)
        if cleaned:
            string += " ".join(cleaned) + " "

    return string.strip()


create_directory("./input")

for threshold in [0, 0.5, 1.0, 1.5, 2.0]:

    output_path = "./input/data_%s.csv" % str(threshold).replace(".", "")

    # topics processing
    with open("./data/topics.csv", "r") as topics_in, open("./data/dataset_text.txt", "r") as labels_in, open(
            output_path, "w") as out:
        # topics csv reader
        topic_reader = csv.reader(topics_in)
        # csv writer
        writer = csv.writer(out)

        # write header
        writer.writerow(["topic", "label"])

        # skip topics csv header
        next(topic_reader, None)

        topics = []
        for row in topic_reader:
            topics.append(clean_word(row[2:]))

        # read labels lines
        lines = labels_in.readlines()
        # append dummy line to write last sequence
        lines.append(DUMMY_LINE)

        last_index = 0

        for line in lines:
            line = line.split("\t")

            # index parsing
            index = int(float(line[0]))
            # new sequence
            if index != last_index:
                last_index = index

            # score parsing
            score = float(line[2].strip("\n"))
            # keep relevant labels only
            if score >= threshold:
                writer.writerow([topics[index], clean_word(line[1].split())])
