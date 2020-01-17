"""Tools for training, evaluating and using POS models."""

import json
import os
import re
from itertools import product

import _jsonnet
import numpy as np
import pandas as pd
from allennlp.predictors import Predictor
import torch

from filenames import POS_CONFIG, POS_MODELS, PROCESSED_POS_DATA
from preprocessing import K, preprocess, tokenize_words

TMP_FILENAME = "tmp.jsonnet"
TRAIN_CMD = "allennlp train -s {directory} -f {config} && rm {config}"
GPU_AVAILABLE = str(torch.cuda.is_available()).lower()

def make_serialization_dirname(options):
    return POS_MODELS.format(**options)


def prepare_config(options):
    with open(POS_CONFIG) as file:
        config_str = file.read()
    for key, value in options.items():
        pattern = f"local {key} = [a-z0-9]+;"
        repl = f"local {key} = {value};"
        config_str = re.sub(pattern, repl, config_str)
    return config_str


# Train


def train(options):
    config_str = prepare_config(options)
    config = json.loads(_jsonnet.evaluate_snippet("snippet", config_str))
    # The override flag in allennlp was finicky so I used a temporary file hack
    with open(TMP_FILENAME, "w") as file:
        json.dump(config, file, indent=2)
    serialization_dir = make_serialization_dirname(options)
    cmd = TRAIN_CMD.format(directory=serialization_dir, config=TMP_FILENAME)
    os.system(cmd)


def train_ensemble(options):
    for k in range(K):
        options["FOLD"] = k
        train(options)


# Predict


def load_model(options):
    serialization_dir = make_serialization_dirname(options)
    filename = os.path.join(serialization_dir, "model.tar.gz")
    return Predictor.from_path(filename, predictor_name="sentence-tagger")


def predict(options, text):
    model = load_model(options)
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


# def predict_ensemble(text):
#     df = pd.DataFrame()
#     for k in range(K):
#         prediction = predict(k, text)
#         df[k] = prediction["tag"]
#     mode = df.mode(axis=1)[0]
#     tokens = tokenize_words(text)
#     return pd.DataFrame({"form": tokens, "tag": mode})


if __name__ == "__main__":
    TOKEN_EMBEDDING_DIMS = [10, 25, 50, 100, 200, 300]
    CHAR_EMBEDDING_DIMS = [5, 10, 20]
    HIDDEN_SIZES = [25, 50, 100, 200]
    BATCH_SIZES = [8, 16, 32, 64]
    USE_PRETRAINED = map(lambda s: str(s).lower(), [True, False])
    FOLDS = range(K)
    for hyperparams in product(
        TOKEN_EMBEDDING_DIMS,
        CHAR_EMBEDDING_DIMS,
        HIDDEN_SIZES,
        BATCH_SIZES,
        USE_PRETRAINED,
        FOLDS,
    ):
        options = {
            "TOKEN_EMBEDDING_DIM": hyperparams[0],
            "CHAR_EMBEDDING_DIM": hyperparams[1],
            "HIDDEN_SIZE": hyperparams[2],
            "BATCH_SIZE": hyperparams[3],
            "USE_PRETRAINED_WORDS": hyperparams[4],
            "USE_PRETRAINED_CHARS": hyperparams[4],
            "NUM_EPOCHS": 10,
            "USE_GPU": GPU_AVAILABLE,
            "FOLD": hyperparams[5],
        }
        # train(options)
