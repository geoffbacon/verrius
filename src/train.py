"""Train models."""

import json
import os

import _jsonnet

from filenames import CONFIG_FILENAME

TRAIN_CMD = "allennlp train -s {directory} -f {config}"
K = 5


def train():
    config = json.loads(_jsonnet.evaluate_file(CONFIG_FILENAME))
    for n in range(K):
        c = config.copy()
        c["train_data_path"] = f"data/evalatin/processed/pos/{n}-train.txt"
        c["validation_data_path"] = f"data/evalatin/processed/pos/{n}-valid.txt"
        serialization_dir = f"models/pos/{n}"
        # The -o override flag in allennlp train was finicky so I used a temporary file hack
        with open("tmp.jsonnet", "w") as file:
            json.dump(c, file, indent=2)
        cmd = TRAIN_CMD.format(config="tmp.jsonnet", directory=serialization_directory)
        os.system(cmd)
        cmd = "rm tmp.jsonnet"
        os.system(cmd)
