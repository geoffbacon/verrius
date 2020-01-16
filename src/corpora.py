import re
from string import punctuation

from cltk.corpus.readers import get_corpus_reader
from cltk.corpus.utils.importer import CorpusImporter
from tqdm import tqdm

from filenames import EXTERNAL_CORPUS_FILENAME
from preprocessing import preprocess
from utils import write

PUNCTUATION = punctuation.replace(".", "")


CORPUS_NAMES = ["latin_text_perseus", "latin_text_tesserae", "latin_text_latin_library"]


def download():
    importer = CorpusImporter("latin")
    for name in CORPUS_NAMES:
        importer.import_corpus(name)


def clean(text):
    """Preprocess `text` like the EvaLatin organizers did."""
    text = "".join([ch for ch in text if ch not in PUNCTUATION])
    text = text.replace("v", "u")
    text = text.replace("j", "i")
    return text


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
            cleaned_sentence = preprocess(clean(sentence))
            cleaned_sentence = re.sub(r"\s+", " ", cleaned_sentence)
            if len(cleaned_sentence.split()) >= 5:
                lines.append(cleaned_sentence)
        except:
            continue
    return lines


if __name__ == "__main__":
    lines = []
    for name in CORPUS_NAMES:
        lines.extend(extract(name))
    write(lines, EXTERNAL_CORPUS_FILENAME)
