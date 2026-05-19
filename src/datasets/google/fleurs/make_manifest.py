"""
This script creates three files, test_manifest.json, train_manifest.json, and
valid_manifest.json, in the data/nexdata/filipino_822 directory. The directory
structure is as follows:

data/
└── google/
    └── fleurs/
        └── fil_ph/
            ├── audio/
            │   ├── dev.tar.gz
            │   ├── test.tar.gz
            │   └── train.tar.gz
            ├── dev.tsv
            ├── test.tsv
            ├── train.tsv
            ├── test_manifest.json
            ├── train_manifest.json
            └── valid_manifest.json
"""

import os
from pathlib import Path

import librosa
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

RANDOM_STATE = 339

CHECKPOINT = "models/g2p/checkpoint-9670"
BATCH_SIZE = 16
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
model = T5ForConditionalGeneration.from_pretrained(CHECKPOINT).to(device)


def get_wav_duration(path):
    return librosa.get_duration(path=str(path))


def predict_batch(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(
        device
    )
    outputs = model.generate(
        **inputs,
        max_length=256,
        num_beams=5,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def make_manifest(type):
    tsv_directory = Path("data/google/fleurs/fil_ph/")

    raw_data = pd.read_csv(
        tsv_directory / (type + ".tsv"),
        sep="\t",
        header=None,
        names=[
            "id",
            "audio",
            "sentence",
            "words_only",
            "words_formatted",
            "sampling_rate",
            "gender",
        ],
    )

    initial_count = len(raw_data)
    raw_data = raw_data[~raw_data["sentence"].astype(str).str.contains(r"\d", na=False)]
    print(f"Filtered out {initial_count - len(raw_data)} rows containing numbers.")

    wav_files = [
        tsv_directory / "audio" / type / filename for filename in raw_data["audio"]
    ]

    texts = raw_data["sentence"].astype(str).tolist()
    predictions = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"G2P {type}"):
        batch = texts[i : i + BATCH_SIZE]
        predictions.extend(predict_batch(batch))

    data = pd.DataFrame(
        {
            "audio_filepath": wav_files,
            "text": predictions,
            "duration": [get_wav_duration(f) for f in wav_files],
        }
    )

    data["text"] = (
        data["text"]
        # IPA does not use standard punctuation
        .str.replace(",", "")
        .str.replace(".", "")
        # Hyphenated words in Tagalog are often read with a glottal stop
        # NOTE: Only replace if not already followed by the glottal stop mark
        .str.replace(r"-(?!ʔ)", "ʔ", regex=True)
        .str.replace("-", "", regex=False)
    )

    print(data)

    manifest_name = "valid" if type == "dev" else type
    data.to_json(
        Path(tsv_directory) / (manifest_name + "_manifest.json"),
        orient="records",
        lines=True,
        default_handler=str,
    )


make_manifest("train")
make_manifest("test")
make_manifest("dev")
