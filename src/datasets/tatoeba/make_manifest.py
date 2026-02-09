import pandas as pd
import os

from datasets import load_dataset
from epitran.backoff import Backoff

# Make sure we have English as a fallback in case it doesn't recognize a word
epi = Backoff(["tgl-Latn", "eng-Latn"])

# Load dataset from Tatoeba
dataset = load_dataset("tatoeba", lang1="en", lang2="tl")

# Convert into IPA using Epitran
sentences = pd.DataFrame(dataset['train']['translation'])
sentences['ipa'] = [epi.transliterate(sentence) for sentence in sentences['tl']] 

print(sentences.head(10))

os.makedirs("data/tatoeba/", exist_ok=True)

sentences[['tl', 'ipa']].to_csv("data/tatoeba/train_manifest.csv")
