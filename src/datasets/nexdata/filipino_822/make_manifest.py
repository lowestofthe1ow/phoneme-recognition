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

import pandas as pd
from epitran.backoff import Backoff

from src.utils.file_read import get_wav_duration, read_file
from src.utils.train_test_val_split import train_test_val_split


def make_manifest(_directory):
    """Creates a manifest file for the NexData 822-hour Filipino speech corpus"""
    directory = Path(_directory)

    g2p = G2P()

    wav_files = [
        directory / file for file in os.listdir(directory) if file.endswith(".wav")
    ]

    txt_files = [file for file in os.listdir(directory) if file.endswith(".txt")]

    # Create the needed data for the manifest file
    data = pd.DataFrame(
        {
            "audio_filepath": sorted(wav_files),
            "text": [
                g2p.transliterate(read_file(directory / filename).strip())
                for filename in sorted(txt_files)
            ],
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

    return data


data = pd.concat(
    [
        make_manifest("data/nexdata/filipino_822/G00001"),
        make_manifest("data/nexdata/filipino_822/G00608"),
    ]
)

train_test_val_split(data, "data/nexdata/filipino_822/")
