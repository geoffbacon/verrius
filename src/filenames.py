"""File name constants used across different modules."""

remote = False
if remote:
    ROOT = "/home/bacon/verrius"
else:
    ROOT = "/Volumes/GoogleDrive/My Drive/research/evalatin/verrius"
RAW_EVALATIN_TRAINING_DATA_DIR = "data/evalatin/raw"
CONFIG_FILENAME = "src/pos.jsonnet"
