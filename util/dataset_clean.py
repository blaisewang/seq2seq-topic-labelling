import csv

SCORE_THRESHOLD = 1.5

DUMMY_LINE = "228.0	#	0.0"

# topics processing
with open("../data/topics.csv", "r") as topics_in, open("../data/topics_cleaned.csv", "w") as topics_out:
    # csv reader
    reader = csv.reader(topics_in)
    # csv writer
    writer = csv.writer(topics_out)
    # skip header
    next(reader, None)

    for row in reader:
        writer.writerow(row[2:])

# labels processing
with open("../data/dataset_text.txt", "r") as labels_in, open("../data/labels_cleaned.csv", "w") as labels_out:
    # read lines
    lines = labels_in.readlines()
    # append dummy line to write last sequence
    lines.append(DUMMY_LINE)
    # csv writer
    writer = csv.writer(labels_out)

    last_index = 0
    label_list = []

    for line in lines:
        line = line.split("\t")

        # index parsing
        index = int(float(line[0]))
        # new sequence
        if index != last_index:
            writer.writerow(label_list)
            last_index = index
            label_list = []

        # score parsing
        score = float(line[2].strip("\n"))
        # keep relevant labels only
        if score >= SCORE_THRESHOLD:
            label_list.append(line[1])
