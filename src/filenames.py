"""File name constants used across different modules."""

import sys

remote = sys.platform == "linux"
if remote:
    ROOT = "/home/bacon/verrius"
else:
    ROOT = "/Volumes/GoogleDrive/My Drive/research/evalatin/verrius"  # for Jupyter notebooks
RAW_EVALATIN_DATA = "data/evalatin/raw"
PROCESSED_POS_DATA = "data/evalatin/processed/pos"
PROCESSED_LEMMA_DATA = "data/evalatin/processed/lemma"
GRAPHEME_PROFILE = "src/profile.txt"
EXTERNAL_CORPUS_FILENAME = "data/external/corpus.txt"
VECTORS_FILENAME = "models/embeddings/{unit}/vectors-{size}.txt"
POS_CONFIG = "src/pos.jsonnet"
POS_MODELS = "models/pos/{FOLD}-{TOKEN_EMBEDDING_DIM}-{CHAR_EMBEDDING_DIM}-{HIDDEN_SIZE}-{BATCH_SIZE}-{USE_PRETRAINED_WORDS}-{USE_PRETRAINED_CHARS}"
LEMMA_MODELS = "models/lemma/{FOLD}.csv"
LOG_DIR = "logs/"
