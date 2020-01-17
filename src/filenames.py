"""File name constants used across different modules."""

ROOT = "/Volumes/GoogleDrive/My Drive/research/evalatin/verrius"  # for Jupyter notebooks
RAW_EVALATIN_DATA = "data/evalatin/raw"
PROCESSED_POS_DATA = "data/evalatin/processed/pos"
GRAPHEME_PROFILE = "src/profile.txt"
EXTERNAL_CORPUS_FILENAME = "data/external/corpus.txt"
CHAR_VECTORS_FILENAME = "models/embeddings/chars/vectors-{}.txt"
WORD_VECTORS_FILENAME = "models/embeddings/words/vectors-{}.txt"
POS_CONFIG = "src/pos.jsonnet"
POS_MODELS = "models/pos/{FOLD}-{TOKEN_EMBEDDING_DIM}-{CHAR_EMBEDDING_DIM}-{HIDDEN_SIZE}-{BATCH_SIZE}-{USE_PRETRAINED_WORDS}-{USE_PRETRAINED_CHARS}"
LOG_DIR = "logs/"
