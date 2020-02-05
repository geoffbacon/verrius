"""Lemmatization models."""

import os
import re
import pandas as pd
from filenames import PROCESSED_LEMMA_DATA, LEMMA_MODELS
from preprocessing import K

ROMAN_NUMERAL_PATTERN = re.compile(
    r"^M{0,4}(CM|CD|D?C{0,4})(XC|XL|L?X{0,4})(IX|I[VU]|[VU]?I{0,4})$", re.IGNORECASE
)

def train(k):
    existing_model_filename = os.path.join(LEMMA_MODELS.format(FOLD=k))
    if os.path.exists(existing_model_filename):
        seen = pd.read_csv(existing_model_filename)
        seen.set_index(["form", "pos"], inplace=True)
    else:
        filename = os.path.join(PROCESSED_LEMMA_DATA, f"{k}-train.csv")
        data = pd.read_csv(filename)
        seen = data.groupby(["form", "pos"]).agg(lambda s: s.mode())
        seen.to_csv(existing_model_filename)
        seen = pd.read_csv(existing_model_filename)
        seen.set_index(["form", "pos"], inplace=True)
    return seen

def evaluate(k):
    seen = train(k)
    filename = os.path.join(PROCESSED_LEMMA_DATA, f"{k}-valid.csv")
    valid = pd.read_csv(filename)
    count, correct = 0, 0
    seen_by_form = seen.reset_index().set_index("form")
    for _, row in valid.iterrows():
        count += 1
        form = row["form"]
        pos = row["pos"]
        key = (form, pos)
        true = row["lemma"]
        try:
            prediction = seen.loc[key]["lemma"]
        except KeyError:
            try:
                prediction = seen_by_form.loc[form]["lemma"]
                if isinstance(prediction, pd.Series):
                    prediction = prediction.iloc[0]
            except KeyError:
                prediction = row["raw"]
        if ROMAN_NUMERAL_PATTERN.match(form) and pos in ["NUM", "ADJ"]:
            prediction = "numerus_romanus"
        if prediction == true:
            correct += 1
    return round(correct / count * 100, 3)

if __name__ == "__main__":
    results = []
    for k in range(K):
        r = evaluate(k)
        results.append(r)
        print(k, r)
    print(sum(results)/K)

