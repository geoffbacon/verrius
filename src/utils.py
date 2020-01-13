"""Basic utilities used across different modules."""

import fileinput
import pathlib

import pyconll

from filenames import RAW_EVALATIN_DATA

SEED = sum(map(ord, "Latin"))


def read():
    """Read in all data into a single pyconll CoNLL structure."""
    path = pathlib.Path(RAW_EVALATIN_DATA)
    files = fileinput.input(path.glob("*.conllu"))
    return pyconll.unit.conll.Conll(files)


def write(lines, filename):
    """Write `lines` to `filename`."""
    with open(filename, "w") as file:
        file.write("\n".join(lines))
