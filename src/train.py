"""Train models."""

import json
import os

import _jsonnet
import numpy as np

from filenames import CONFIG_FILENAME

TRAIN_CMD = "allennlp train -s {directory} -f {config}"
K = 5


def train():
    config = json.loads(_jsonnet.evaluate_file(CONFIG_FILENAME))
    for k in range(K):
        c = config.copy()
        c["train_data_path"] = f"data/evalatin/processed/pos/{k}-train.txt"
        c["validation_data_path"] = f"data/evalatin/processed/pos/{k}-valid.txt"
        serialization_dir = f"models/pos/{k}"
        # The -o override flag in allennlp train was finicky so I used a temporary file hack
        with open("tmp.jsonnet", "w") as file:
            json.dump(c, file, indent=2)
        cmd = TRAIN_CMD.format(config="tmp.jsonnet", directory=serialization_dir)
        os.system(cmd)
        cmd = "rm tmp.jsonnet"
        os.system(cmd)

def score():
    accuracies = []
    for k in range(K):
        filename = f"models/pos/{k}/metrics.json"
        with open(filename) as file:
            metrics = json.load(file)
            acc = metrics["validation_accuracy"]
            accuracies.append(acc)
    mean = round(np.mean(accuracies), 3)
    std = round(np.std(accuracies), 3)
    print(mean, std)
    print(accuracies)