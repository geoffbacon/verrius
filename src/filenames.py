"""File name constants used across different modules."""

remote = False
if remote:
    ROOT = "/home/bacon/verrius"
else:
    ROOT = "/Volumes/GoogleDrive/My Drive/research/evalatin/verrius"
RAW_EVALATIN_DATA = "data/evalatin/raw"
PROCESSED_POS_DATA = "data/evalatin/processed/pos"
GRAPHEME_PROFILE = "src/profile.txt"
POS_CONFIG = "src/pos.jsonnet"
POS_MODELS = "models/pos"
LOG_DIR = "logs/"
VECTORS_FILENAME_TEMPLATE = "models/embeddings/vectors-{}-{}.txt"
PERSEUS_FILENAME = "data/perseus/processed.txt"
