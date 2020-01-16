# Verrius

Part-of-speech tagger and lemmatizer for Latin.

## To do
- POS tagging
    - Pre-trained embeddings *
        - Train
    - Hyperparameter optimization *
        - 10-fold CV
        - TOKEN_EMBEDDING_DIM
            - 10, 25, 50, 100, 200, 300
        - CHAR_EMBEDDING_DIM
            - 5, 10
        - BATCH_SIZE
            - 16, 32
        - BIDIRECTIONAL_LSTM["hidden_size"]
            - 50, 100, 200
        - Word embeddings
            - With and without
    - Postprocessing
    - Evaluation (output and script)
    - Error analysis
    - Facilitate engagement
    - Lint
    - Preprocessing
        - Start/end sentence boundaries
- Lemmatization
- External unlabelled data
- External labelled data
- Data augmentation methods

## Current score
0.901 0.005