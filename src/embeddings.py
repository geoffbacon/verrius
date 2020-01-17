"""Train embeddings."""
import logging
import os
import warnings

from gensim.models import FastText
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.keyedvectors import FastTextKeyedVectors
from tqdm import tqdm

from filenames import EXTERNAL_CORPUS_FILENAME, LOG_DIR, WORD_VECTORS_FILENAME, CHAR_VECTORS_FILENAME
from preprocessing import WORD_SEPARATOR, WORD_TAG_DELIMITER

# silence gensim's logging
logging.getLogger("gensim").setLevel(logging.ERROR)
warnings.simplefilter(action="ignore", category=UserWarning)


class Corpus:
    """Provide access to external unlabelled data."""

    def __init__(self, unit="words"):
        self.unit = unit

    def __iter__(self):
        with open(EXTERNAL_CORPUS_FILENAME) as file:
            for line in file:
                if self.unit == "words":
                    yield line.strip().split()
                elif self.unit == "chars":
                    for word in line.strip().split():
                        yield list(word)

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
        logging.info(f"Training size={model.vector_size}")

    def on_train_end(self, model):
        logging.info(f"Finished training size={model.vector_size}")


# pylint: disable=C0330
def train(
    unit="words",  # word or char embeddings
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
    if unit == "chars":
        assert ngrams == 0, "Can't use fasttext on characters"
    corpus = Corpus(unit=unit)
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
    model.build_vocab(sentences=corpus)
    callback = Callback(epochs)
    model.train(
        sentences=corpus,
        total_examples=num_lines,
        total_words=num_words,
        epochs=epochs,
        callbacks=[callback],
    )
    if unit == "words":
        filename = WORD_VECTORS_FILENAME.format(size)
    elif unit == "chars":
        filename = CHAR_VECTORS_FILENAME.format(size)
    model.wv.save_word2vec_format(filename, binary=False)
    # Remove header line
    with open(filename, "r") as file:
        lines = file.readlines()
    with open(filename, "w") as file:
        file.write("".join(lines[1:]))


# if __name__ == "__main__":
#     for size in [10, 25, 50, 100, 200, 300]:
#         train(size=size)
