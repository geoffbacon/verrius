"""File name constants used across different modules."""

import sys

remote = sys.platform == "linux"
if remote:
    ROOT = "/home/bacon/verrius"
else:
    ROOT = "/Volumes/GoogleDrive/My Drive/research/evalatin/verrius"
RAW_EVALATIN_DATA = "data/evalatin/raw"
PROCESSED_POS_DATA = "data/evalatin/processed/pos"
GRAPHEME_PROFILE = "src/profile.txt"
POS_CONFIG = "src/pos.jsonnet"
POS_MODELS = "models/pos/{FOLD}-{TOKEN_EMBEDDING_DIM}-{CHAR_EMBEDDING_DIM}-{HIDDEN_SIZE}-{BATCH_SIZE}-{USE_PRETRAINED_WORDS}-{USE_PRETRAINED_CHARS}"
LOG_DIR = "logs/"
WORD_VECTORS_FILENAME = "models/embeddings/words/vectors-{}.txt"
CHAR_VECTORS_FILENAME = "models/embeddings/chars/vectors-{}.txt"
EXTERNAL_CORPUS_FILENAME = "data/external/corpus.txt"
