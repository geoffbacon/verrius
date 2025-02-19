# Configuration file for POS tagger

## Hyperparams
local TOKEN_EMBEDDING_DIM = 100;
local CHAR_EMBEDDING_DIM = 10;
local HIDDEN_SIZE = 100;
local BATCH_SIZE = 32;
local USE_PRETRAINED_WORDS = false;
local USE_PRETRAINED_CHARS = false;
local NUM_EPOCHS = 10;

# Other options
local USE_GPU = false;
local FOLD = 0;
local PRETRAINED_WORDS = "models/embeddings/words/vectors-" + TOKEN_EMBEDDING_DIM + ".txt";
local PRETRAINED_CHARS = "models/embeddings/chars/vectors-" + CHAR_EMBEDDING_DIM + ".txt";
local TRAIN_DATA_PATH = "data/evalatin/processed/pos/" + FOLD + "-train.txt";
local VALID_DATA_PATH ="data/evalatin/processed/pos/" + FOLD + "-valid.txt";

local CHARACTER_LSTM = {
    "type": "lstm",
    "input_size": CHAR_EMBEDDING_DIM,
    "hidden_size": CHAR_EMBEDDING_DIM,
    "num_layers": 1,
    "bidirectional": false
};

local BIDIRECTIONAL_LSTM = {
    "type": "lstm",
    "hidden_size": HIDDEN_SIZE,
    "input_size": TOKEN_EMBEDDING_DIM + CHARACTER_LSTM["hidden_size"],
    "dropout": 0.5,
    "num_layers": 2,
    "bidirectional": true
};

## Static
{
    "dataset_reader": {
        "type": "sequence_tagging",
        "word_tag_delimiter": "/",
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": false
            },
            "token_characters": {
                "type": "characters",
                "min_padding_length": 1
            },
        },
    },
    "train_data_path": TRAIN_DATA_PATH,
    "validation_data_path": VALID_DATA_PATH,
    "iterator": {
        "type": "basic",
        "batch_size": BATCH_SIZE
    },
    "trainer": {
        "optimizer": {
            "type": "adam"
        },
        "num_epochs": NUM_EPOCHS,
        "patience": 2,
        "cuda_device": if USE_GPU then 0 else -1,
        "shuffle": true,
        "num_serialized_models_to_keep": -1
    },
    "model": {
        "type": "simple_tagger",
        "text_field_embedder": {
            "type": "basic",
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": TOKEN_EMBEDDING_DIM,
                    "trainable": true,
                    "pretrained_file": if USE_PRETRAINED_WORDS then PRETRAINED_WORDS else ""
                },
                "token_characters": {
                    "type": "character_encoding",
                    "embedding": {
                        "embedding_dim": CHAR_EMBEDDING_DIM,
                        "trainable": true,
                        "pretrained_file": if USE_PRETRAINED_CHARS then PRETRAINED_CHARS else ""
                    },
                    "encoder": CHARACTER_LSTM,
                    "dropout": 0.1
                }
            }
        },
        "encoder": BIDIRECTIONAL_LSTM
    }
}