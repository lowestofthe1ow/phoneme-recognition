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

import epitran
import pandas as pd
from epitran.backoff import Backoff

from src.utils.file_read import get_wav_duration, read_file

RANDOM_STATE = 339


def make_manifest(type):
    tsv_directory = Path("data/google/fleurs/fil_ph/")

    epi = Backoff(["tgl-Latn", "eng-Latn"])

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

    wav_files = [
        tsv_directory / "audio" / type / filename for filename in raw_data["audio"]
    ]

    print(wav_files)

    data = pd.DataFrame(
        {
            "audio_filepath": wav_files,
            "text": [epi.transliterate(word) for word in raw_data["words_only"]],
            "duration": [get_wav_duration(filename) for filename in sorted(wav_files)],
        }
    )

    data["text"] = (
        data["text"]
        # IPA does not use standard punctuation
        .str.replace(",", "")
        .str.replace("'", "")
        .str.replace(".", "")
        # Hyphenated words in Tagalog are often read with a glottal stop
        # NOTE: Only replace if not already followed by the glottal stop mark
        .str.replace(r"-(?!ʔ)", "ʔ", regex=True)
        .str.replace("-", "", regex=False)
    )

    print(data)

    data.to_json(
        Path(tsv_directory) / (type + "_manifest.json"),
        orient="records",
        lines=True,
        default_handler=str,
    )


make_manifest("train")
make_manifest("test")
make_manifest("dev")
