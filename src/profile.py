"""Create orthography profile for grapheme tokenization."""

from collections import OrderedDict

from segments import Profile

from filenames import GRAPHEME_PROFILE
from preprocessing import clean
from utils import read

# Read in all data into a single pyconll CoNLL structure
conll = read()

# Collect all the word forms
text = ""
for sentence in conll:
    for token in sentence:
        text += clean(token.form) + " "

# Create orthography profile
profile = Profile.from_text(text)
profile.column_labels.remove("frequency")
profile.graphemes.pop(" ")
for key in ["ch", "qu", "th", "rh", "ph", "gn"]:
    profile.graphemes[key] = OrderedDict([("mapping", key[0].upper())])
    profile.graphemes.move_to_end(key, last=False)
with open(GRAPHEME_PROFILE, "w") as file:
    file.write(str(profile))
