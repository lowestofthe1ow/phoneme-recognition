"""
This script creates three files, test_manifest.json, train_manifest.json, and
valid_manifest.json, in the data/nexdata/filipino_822 directory. The directory
structure is as follows:

data/
├── nexdata/
│   └── filipino_822/
│       ├── G00001/
│       ├── G00608/
│       ├── test_manifest.json
│       ├── train_manifest.json
│       └── valid_manifest.json
└── samples/
"""

import os
from pathlib import Path

import epitran
import pandas as pd
from epitran.backoff import Backoff
from sklearn.model_selection import train_test_split

from src.utils.file_read import get_wav_duration, read_file

RANDOM_STATE = 339


def make_manifest(_directory):
    """Creates a manifest file for the NexData 822-hour Filipino speech corpus"""
    directory = Path(_directory)

    # epi = epitran.Epitran("tgl-Latn")
    epi = Backoff(["tgl-Latn", "eng-Latn"])

    wav_files = [
        directory / file for file in os.listdir(directory) if file.endswith(".wav")
    ]

    txt_files = [file for file in os.listdir(directory) if file.endswith(".txt")]

    # Create the needed data for the manifest file
    data = pd.DataFrame(
        {
            "audio_filepath": sorted(wav_files),
            "text": [
                epi.transliterate(read_file(directory / filename).strip())
                for filename in sorted(txt_files)
            ],
            "duration": [get_wav_duration(filename) for filename in sorted(wav_files)],
        }
    )

    return data


data = pd.concat(
    [
        make_manifest("data/nexdata/filipino_822/G00001"),
        make_manifest("data/nexdata/filipino_822/G00608"),
    ]
)

# Initial split: "train"/test (90% vs 10%)
all_train_df, test_df = train_test_split(
    data, test_size=0.1, random_state=RANDOM_STATE, shuffle=True
)

print(f"Total train data duration: {all_train_df['duration'].sum()}")

# Second split: train/valid (80% / 10% (of the whole dataset))
train_df, val_df = train_test_split(
    all_train_df, test_size=0.1 / 0.9, random_state=RANDOM_STATE, shuffle=True
)

print(f"Original data shape: {data.shape}")
print(f"Train split shape: {train_df.shape}")
print(f"Validation split shape: {val_df.shape}")
print(f"Test split shape: {test_df.shape}")

train_df.to_json(
    "data/nexdata/filipino_822/train_manifest.json",
    orient="records",
    lines=True,
    default_handler=str,
)

val_df.to_json(
    "data/nexdata/filipino_822/valid_manifest.json",
    orient="records",
    lines=True,
    default_handler=str,
)

test_df.to_json(
    "data/nexdata/filipino_822/test_manifest.json",
    orient="records",
    lines=True,
    default_handler=str,
)
