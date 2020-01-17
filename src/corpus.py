"""Create unlabelled corpus of Latin to learn embeddings from.

This module is intended to be run as a script:
    $ python src/corpus.py
    
"""

import re

from cltk.corpus.readers import get_corpus_reader
from cltk.corpus.utils.importer import CorpusImporter
from tqdm import tqdm

from filenames import EXTERNAL_CORPUS_FILENAME
from preprocessing import preprocess, preprocess_like_evalatin
from utils import write

CORPUS_NAMES = ["latin_text_perseus", "latin_text_tesserae", "latin_text_latin_library"]


def download(names):
    importer = CorpusImporter("latin")
    for name in names:
        importer.import_corpus(name)


def extract(name):
    reader = get_corpus_reader(language="latin", corpus_name=name)
    lines = []
    if name == "latin_text_perseus":
        sentences = reader.sents()
    elif name == "latin_text_tesserae":
        sentences = reader.sents(fileids=reader.fileids())
    elif name == "latin_text_latin_library":
        sentences = (" ".join(sentence) for sentence in reader.sents())
    for sentence in tqdm(sentences):
        try:
            cleaned_sentence = preprocess(preprocess_like_evalatin(sentence))
            cleaned_sentence = re.sub(r"\s+", " ", cleaned_sentence).strip()
            if len(cleaned_sentence.split()) >= 5:
                if "�" not in cleaned_sentence:
                    lines.append(cleaned_sentence)
        except:
            continue
    return lines


if __name__ == "__main__":
    download(CORPUS_NAMES)
    lines = []
    for name in CORPUS_NAMES:
        lines.extend(extract(name))
    write(lines, EXTERNAL_CORPUS_FILENAME)
