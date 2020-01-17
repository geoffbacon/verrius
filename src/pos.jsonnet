# Configuration file for POS tagger

## Hyperparams
local TOKEN_EMBEDDING_DIM = 100;
local CHAR_EMBEDDING_DIM = 10;
local HIDDEN_SIZE = 100;
local BATCH_SIZE = 32;
local USE_PRETRAINED = false;
local PRETRAINED_FILE = "models/embeddings/vectors-" + TOKEN_EMBEDDING_DIM + ".txt";
local FOLD = 0;
local NUM_EPOCHS = 10;
local USE_GPU = false;

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
    "train_data_path": "data/evalatin/processed/pos/" + FOLD + "-train.txt",
    "validation_data_path": "data/evalatin/processed/pos/" + FOLD + "-valid.txt",
    "iterator": {
        "type": "basic",
        "batch_size": BATCH_SIZE
    },
    "trainer": {
        "optimizer": {
            "type": "adam"
        },
        "num_epochs": NUM_EPOCHS,
        "patience": std.max(std.floor(NUM_EPOCHS / 2), 2),
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
                    "pretrained_file": if USE_PRETRAINED then PRETRAINED_FILE else ""
                },
                "token_characters": {
                    "type": "character_encoding",
                    "embedding": {
                        "embedding_dim": CHAR_EMBEDDING_DIM,
                        "trainable": true
                    },
                    "encoder": CHARACTER_LSTM,
                    "dropout": 0.1
                }
            }
        },
        "encoder": BIDIRECTIONAL_LSTM
    }
}