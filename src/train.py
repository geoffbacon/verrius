"""Train models."""

import json
import os

import _jsonnet

from filenames import POS_CONFIG_FILENAME

TRAIN_CMD = "allennlp train -s {directory} -f {config}"


def train():
    train_data_path = "data/evalatin/processed/pos.txt"
    serialization_directory = "models"
    # The -o override flag in allennlp train was finicky so I used a temporary file hack
    config = json.loads(_jsonnet.evaluate_file(POS_CONFIG_FILENAME))
    config["train_data_path"] = train_data_path
    with open("tmp.jsonnet", "w") as file:
        json.dump(config, file, indent=2)
    cmd = TRAIN_CMD.format(config="tmp.jsonnet", directory=serialization_directory)
    os.system(cmd)
    cmd = "rm tmp.jsonnet"
    os.system(cmd)
