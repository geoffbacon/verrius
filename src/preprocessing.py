"""Preprocessing Latin text data and preparing it for POS tagging and lemmatization."""

# I replace some types of forms (e.g. Greek words) with placeholder markers because
# the specifics of their forms don't matter for these tasks. I also insert start and
# end markers for word and sentence boundaries. For all these, I want the marker to
# be a single token for convenience which do not appear in the cleaned text, and so
# have chosen Greek letters for this task. Prior to adding any of these markers,
# Greek letters from the original text have been removed.

import re
import unicodedata

from segments import Tokenizer

GREEK_TOKEN = "α"
LACUNA_TOKEN = "β"
PROPN_ABBREVIATION_TOKEN = "γ"
START_WORD = "δ"
END_WORD = "ε"
START_SENTENCE = "ζ"
END_SENTENCE = "η"
GRAPHEME_SEPARATOR = ""
WORD_DELIMITER = "\t"

char_class = lambda ch: unicodedata.name(ch).split()[0]


def remove_other_chars(word):
    return "".join([ch for ch in word if char_class(ch) in ["LATIN", "GREEK", "FULL"]])


def is_greek_char(ch):
    return char_class(ch) == "GREEK"


def is_greek_word(word):
    return any(map(is_greek_char, word))


def replace_greek_word(word):
    if is_greek_word(word):
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
    word = word.lower()  # might not want to do this
    return word


tokenizer = Tokenizer("src/profile.prf")


def preprocess(text):
    result = []
    for word in text.split():
        form = tokenizer(
            clean(word), segment_separator=GRAPHEME_SEPARATOR, column="mapping"
        )
        form = GRAPHEME_SEPARATOR.join(
            [START_WORD, form, END_WORD]
        )  # add in start/end word boundaries
        result.append(form)
    result[0] = START_SENTENCE + GRAPHEME_SEPARATOR + result[0]
    result[-1] = result[-1] + GRAPHEME_SEPARATOR + END_SENTENCE
    return WORD_DELIMITER.join(result)
