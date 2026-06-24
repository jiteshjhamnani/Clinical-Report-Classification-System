# create_medical_terms.pkl

import pandas as pd
import pickle

df = pd.read_csv("dataset/mtsamples.csv")

medical_terms = set()

for row in df["keywords"].dropna():

    terms = str(row).lower().split(",")

    for term in terms:

        term = term.strip()

        if len(term) > 2:
            medical_terms.add(term)

print("Total Terms:", len(medical_terms))

with open("medical_terms.pkl", "wb") as f:
    pickle.dump(medical_terms, f)

print("Saved Successfully")