"""Stuff to do with models"""

import json
import os
from functools import reduce

import _jsonnet
import numpy as np
import pandas as pd
from allennlp.predictors import Predictor

from filenames import CONFIG_FILENAME
from preprocessing import preprocess

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


def load_model(path):
    filename = os.path.join(path, "model.tar.gz")
    return Predictor.from_path(filename, predictor_name="sentence-tagger")


def predict(model, text):
    preprocessed_text = preprocess(text)
    result = model.predict(preprocessed_text)
    labels = model._model.vocab.get_token_to_index_vocabulary("labels")
    labels = sorted(labels, key=labels.__getitem__)
    forms = text.split()
    tags = result["tags"]
    tmp = pd.DataFrame({"form": forms, "tag": tags})
    df = pd.DataFrame(result["class_probabilities"], columns=labels)
    result = pd.merge(tmp, df, left_index=True, right_index=True)
    return result


def ensemble_predict(text):
    dfs = []
    for k in range(K):
        path = f"models/pos/{k}"
        model = load_model(path)
        result = predict(model, text)
        dfs.append(result[["form", "tag"]])
    merged = reduce(
        lambda left, right: pd.merge(left, right, left_index=True, right_index=True),
        dfs,
    )
    columns = [column for column in merged.columns if "tag" in column]
    mode = merged[columns].mode(axis=1)[0]
    result = pd.DataFrame({"form": merged["form_x"].iloc[:, 0], "tag": mode})
    return result


model = load_model("models/pos/0")
text = "sed hoc homo qui currit est pater"
result = ensemble_predict(text)
