"""Preprocessing tools for Latin text."""

import os
import re
import unicodedata
from string import punctuation

import fire
from segments import Tokenizer
from sklearn.model_selection import KFold

from filenames import GRAPHEME_PROFILE, PROCESSED_POS_DATA
from utils import SEED, read, write

PUNCTUATION = punctuation.replace(".", "")

# Clean word forms

# We replace some types of word forms (e.g. Greek words) with placeholder
# characters because their specific forms don't matter for our tasks. We also
# add start and end characters for word and sentence boundaries. For all of
# these additional values, it's easiest to use single characters that don't
# appear in the text. As we've remove the Greek words, we chose to use Greek
# characters as our placeholders. These are purely internal to this codebase.

GREEK_TOKEN = "α"
LACUNA_TOKEN = "β"
PROPN_ABBREVIATION_TOKEN = "γ"
START_WORD = "δ"
END_WORD = "ε"
START_SENTENCE = "ζ"
END_SENTENCE = "η"
GRAPHEME_SEPARATOR = ""
WORD_SEPARATOR = " "
WORD_TAG_DELIMITER = "/"


def char_class(ch):
    return unicodedata.name(ch).split()[0]


def remove_other_chars(word):
    chars = [ch for ch in word if char_class(ch) in ["LATIN", "GREEK", "FULL"]]
    return "".join(chars)


def replace_greek_word(word):
    for ch in word:
        if char_class(ch) == "GREEK":
            return GREEK_TOKEN
    return word


def replace_salus(word):
    if word == "s.":
        return "salus"
    return word


def replace_lacuna(word):
    match = re.search(r"\.\.", word)
    if match:
        return LACUNA_TOKEN
    return word


def replace_propn_abbreviation(word):
    match = re.match(r"[A-Z].*\.", word)
    if match:
        return PROPN_ABBREVIATION_TOKEN
    return word


def replace_full_stop(word):
    return word.replace(".", "")


def replace_j(word):
    return word.replace("j", "")


def clean(word):
    word = remove_other_chars(word)
    word = replace_greek_word(word)
    word = replace_salus(word)
    word = replace_lacuna(word)
    word = replace_propn_abbreviation(word)
    word = replace_full_stop(word)
    word = replace_j(word)
    word = word.lower()
    return word


# Tokenize

tokenize_graphemes = Tokenizer(GRAPHEME_PROFILE)


def clean_and_tokenize(word):
    word = clean(word)
    graphemes = tokenize_graphemes(
        word, segment_separator=GRAPHEME_SEPARATOR, column="mapping"
    )
    if graphemes:
        return GRAPHEME_SEPARATOR.join([START_WORD, graphemes, END_WORD])
    return ""


def tokenize_words(text):
    return text.split()


# Preprocessing

def preprocess_like_evalatin(text):
    """Preprocess `text` like the EvaLatin organizers did, before my own preprocessing.
    
    We only need this for `text` that doesn't come from the EvaLatin organizers.
    
    """
    text = "".join([ch for ch in text if ch not in PUNCTUATION])
    text = text.replace("v", "u")
    text = text.replace("j", "i")
    return text


def preprocess(text):
    """Preprocess unlabelled text."""
    words = tokenize_words(text)
    words = [clean_and_tokenize(word) for word in words]
    return WORD_SEPARATOR.join(words)


# Prepare POS data

K = 10


def prepare_pos(num_splits=K):
    """Prepare data for POS tagging.

    We use K-form cross-validation. To ensure consistency across experiments
    as well as to simplify our implementation, we pre-compute the folds and
    save them to disk.
    """
    # Read in all data into a single pyconll CoNLL structure
    conll = read()

    # Clean, tokenize and prepare each sentence
    data = []
    for sentence in conll:
        line = []
        for token in sentence:
            cleaned_token = clean_and_tokenize(token.form)
            pos = token.upos
            instance = cleaned_token + WORD_TAG_DELIMITER + pos
            line.append(instance)
        data.append(WORD_SEPARATOR.join(line))

    cv = KFold(num_splits, shuffle=True, random_state=SEED)
    for k, (train_idx, valid_idx) in enumerate(cv.split(data)):
        train = [data[i] for i in train_idx]
        valid = [data[i] for i in valid_idx]
        filename = os.path.join(PROCESSED_POS_DATA, f"{k}-train.txt")
        write(train, filename)
        filename = os.path.join(PROCESSED_POS_DATA, f"{k}-valid.txt")
        write(valid, filename)

if __name__ == "__main__":
    fire.Fire()
