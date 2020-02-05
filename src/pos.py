"""Tools for training, evaluating and using POS models."""

import json
import os
import re
from itertools import product

import _jsonnet
import numpy as np
import pandas as pd
import pyconll
import torch
from allennlp.predictors import Predictor

from filenames import POS_CONFIG, POS_MODELS, PROCESSED_POS_DATA
from preprocessing import (K, preprocess, preprocess_like_evalatin,
                           tokenize_words)

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


# Predict


def load_model(serialization_dir):
    filename = os.path.join(serialization_dir, "model.tar.gz")
    return Predictor.from_path(filename, predictor_name="sentence-tagger")


def predict_from_text(model, text):
    preprocessed_text = preprocess(preprocess_like_evalatin(text))
    result = model.predict(preprocessed_text)
    predicted_tags = result["tags"]
    labels = model._model.vocab.get_token_to_index_vocabulary("labels")
    labels = sorted(labels, key=labels.__getitem__)
    tokens = tokenize_words(text)
    df = pd.DataFrame({"form": tokens, "tag": predicted_tags})
    probs = pd.DataFrame(result["class_probabilities"], columns=labels)
    df = pd.merge(df, probs, left_index=True, right_index=True)
    return df


if __name__ == "__main__":
    USE_PRETRAINED = ["true"]
    BATCH_SIZES = [8, 16]
    HIDDEN_SIZES = [25, 50, 100, 200]
    TOKEN_EMBEDDING_DIMS = [10, 25, 50, 100, 200, 300]
    CHAR_EMBEDDING_DIMS = [5, 10, 20]
    FOLDS = range(K)
    for hyperparams in product(
        USE_PRETRAINED,
        BATCH_SIZES,
        HIDDEN_SIZES,
        TOKEN_EMBEDDING_DIMS,
        CHAR_EMBEDDING_DIMS,
        FOLDS,
    ):
        options = {
            "USE_PRETRAINED_WORDS": hyperparams[0],
            "USE_PRETRAINED_CHARS": hyperparams[0],
            "BATCH_SIZE": hyperparams[1],
            "HIDDEN_SIZE": hyperparams[2],
            "TOKEN_EMBEDDING_DIM": hyperparams[3],
            "CHAR_EMBEDDING_DIM": hyperparams[4],
            "FOLD": hyperparams[5],
            "NUM_EPOCHS": 10,
            "USE_GPU": GPU_AVAILABLE,
        }
        # train(options)
