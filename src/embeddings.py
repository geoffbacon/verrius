"""Train embeddings."""
import logging
import os
import warnings

import numpy as np
from gensim.models import FastText
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.keyedvectors import FastTextKeyedVectors
from tqdm import tqdm

from filenames import (LOG_DIR, PERSEUS_FILENAME, PROCESSED_POS_DATA,
                       VECTORS_FILENAME_TEMPLATE)
from preprocessing import WORD_SEPARATOR, WORD_TAG_DELIMITER, K

# silence gensim's logging
logging.getLogger("gensim").setLevel(logging.ERROR)
warnings.simplefilter(action="ignore", category=UserWarning)


class Corpus:
    """Provide access to the training data preprocessed text of a POS-tagged text file."""

    def __iter__(self):
        with open(PERSEUS_FILENAME) as file:
            for line in file:
                yield line.strip().split()

    def count(self):
        """Return the number of lines and words in the corpus."""
        num_lines, num_words = 0, 0
        for line in tqdm(self):
            num_lines += 1
            num_words += len(line)
        return num_lines, num_words


# pylint: disable=W1203
class Callback(CallbackAny2Vec):
    """Simple callback to display progress during training."""

    def __init__(self, epochs):
        super().__init__()
        self.i = 1
        self.epochs = epochs
        logging.basicConfig(
            filename=os.path.join(LOG_DIR, "fasttext.log"),
            format="%(asctime)s : %(levelname)s : %(message)s",
            datefmt="%d %B %H:%M:%S",
            level=logging.INFO,
        )

    def on_epoch_end(self, model):
        logging.info(f"Epoch {self.i} of {self.epochs} ended")
        self.i += 1

    def on_train_begin(self, model):
        logging.info(f"Training K={model._k} size={model.vector_size}")

    def on_train_end(self, model):
        logging.info(f"Finished training K={model._k} size={model.vector_size}")


# pylint: disable=C0330
def train(
    k,  # the cross-validation fold to train on
    size=300,  # size of the embeddings
    window=4,  # context window
    epochs=10,  # number of iterations over the corpus
    min_ngram=3,  # minimum length of n-grams
    max_ngram=6,  # maximum length of n-grams
    min_count=3,  # minimum token frequency
    skipgram=1,  # use skipgram over CBOW
    ngrams=1,  # use fasttext over word2vec
    workers=4,  # number of threads
):
    """Training embeddings.
    
    Hyperparameters can be specified either at the command line.
    
    """
    corpus = Corpus()
    num_lines, num_words = corpus.count()
    model = FastText(
        size=size,
        window=window,
        min_n=min_ngram,
        max_n=max_ngram,
        min_count=min_count,
        sg=skipgram,
        word_ngrams=ngrams,
        workers=workers,
    )
    model._k = "perseus"
    model.build_vocab(sentences=corpus)
    callback = Callback(epochs)
    model.train(
        sentences=corpus,
        total_examples=num_lines,
        total_words=num_words,
        epochs=epochs,
        callbacks=[callback],
    )
    filename = VECTORS_FILENAME_TEMPLATE.format(k, size)
    model.wv.save_word2vec_format(filename, binary=False)
    # Remove header line
    with open(filename, "r") as file:
        lines = file.readlines()
    with open(filename, "w") as file:
        file.write("".join(lines[1:]))


if __name__ == "__main__":
    train("perseus", size=100)
