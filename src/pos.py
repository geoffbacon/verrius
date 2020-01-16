"""POS tagging model"""

import json
import os
import re

import _jsonnet
import numpy as np
import pandas as pd
from allennlp.predictors import Predictor

from filenames import (POS_CONFIG, POS_MODELS, PROCESSED_POS_DATA, VECTORS_FILENAME)
from preprocessing import K, preprocess, tokenize_words

TMP_FILENAME = "tmp.jsonnet"
TRAIN_CMD = "allennlp train -s {directory} -f {config} && rm {config}"

# Prepare config


def prepare(options):
    with open(POS_CONFIG) as file:
        contents = file.read()
    for key, value in options.items():
        pattern = f"local {key} = [a-z0-9]+;"
        repl = f"local {key} = {value};"
        contents = re.sub(pattern, repl, contents)
    with open(TMP_FILENAME, "w") as file:
        file.write(contents)


# Train


def train(k, options=None):
    prepare(options)
    config = json.loads(_jsonnet.evaluate_file(TMP_FILENAME))
    config["train_data_path"] = os.path.join(PROCESSED_POS_DATA, f"{k}-train.txt")
    config["validation_data_path"] = os.path.join(PROCESSED_POS_DATA, f"{k}-valid.txt")
    # The override flag in allennlp was finicky so I used a temporary file hack
    with open(TMP_FILENAME, "w") as file:
        json.dump(config, file, indent=2)
    serialization_dir = os.path.join(POS_MODELS, str(k))
    cmd = TRAIN_CMD.format(directory=serialization_dir, config=TMP_FILENAME)
    os.system(cmd)


def train_ensemble(options=None):
    for k in range(K):
        train(k, options)


# Evaluate


def score(k):
    filename = os.path.join(POS_MODELS, str(k), "metrics.json")
    with open(filename) as file:
        metrics = json.load(file)
        accuracy = round(metrics["validation_accuracy"], 4)
    return accuracy


def score_ensemble():
    accuracies = [score(k) for k in range(K)]
    mean = round(np.mean(accuracies), 3)
    std = round(np.std(accuracies), 3)
    print(mean, std)
    print(accuracies)


# Predict


def load_model(k):
    filename = os.path.join(POS_MODELS, str(k), "model.tar.gz")
    return Predictor.from_path(filename, predictor_name="sentence-tagger")


def predict(k, text):
    model = load_model(k)
    preprocessed_text = preprocess(text)
    result = model.predict(preprocessed_text)
    predicted_tags = result["tags"]
    labels = model._model.vocab.get_token_to_index_vocabulary("labels")
    labels = sorted(labels, key=labels.__getitem__)
    tokens = tokenize_words(text)
    df = pd.DataFrame({"form": tokens, "tag": predicted_tags})
    probs = pd.DataFrame(result["class_probabilities"], columns=labels)
    df = pd.merge(df, probs, left_index=True, right_index=True)
    return df


def predict_ensemble(text):
    df = pd.DataFrame()
    for k in range(K):
        prediction = predict(k, text)
        df[k] = prediction["tag"]
    mode = df.mode(axis=1)[0]
    tokens = tokenize_words(text)
    return pd.DataFrame({"form": tokens, "tag": mode})

# if __name__ == "__main__":
#     options = {"TOKEN_EMBEDDING_DIM": 100,
#                "CHAR_EMBEDDING_DIM": 10,
#                "HIDDEN_SIZE": 100,
#                "BATCH_SIZE": 32,
#                "USE_PRETRAINED": "false"}
#     train(0, options)
