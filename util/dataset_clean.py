import csv
import re

SCORE_THRESHOLD = 1.5

DUMMY_LINE = "228.0	#	-1.0"


def clean_word(words):
    string = ""

    for word in words:

        word = word.replace(".", "")

        cleaned = re.findall(r"[\w/-]+", word)
        if cleaned:
            string += " ".join(cleaned) + " "

    return string.strip()


# topics processing
with open("../data/topics.csv", "r") as topics_in, open("../data/dataset_text.txt", "r") as labels_in, open(
        "../input/data.csv", "w") as out:
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
        if score >= SCORE_THRESHOLD:
            writer.writerow([topics[index], clean_word(line[1].split())])
