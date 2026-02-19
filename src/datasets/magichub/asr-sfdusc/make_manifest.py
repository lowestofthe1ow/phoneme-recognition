"""
This script creates three files, test_manifest.json, train_manifest.json, and
valid_manifest.json, in the data/magichub/asr-sfdusc directory. The directory
structure is as follows:

data/
├── magichub/
│   └── asr-sfdusc/
│       ├── README.txt
│       ├── SPKINFO.txt
│       ├── UTTRANSINFO.txt
│       └── WAV/
└── samples/
"""

from pathlib import Path

import pandas as pd
from epitran.backoff import Backoff

from src.utils.train_test_val_split import train_test_val_split

raw_data = pd.read_csv("data/magichub/asr-sfdusc/UTTRANSINFO.txt", sep="\t")

raw_data["path"] = raw_data["SPEAKER_ID"] + "/" + raw_data["UTTRANS_ID"]

base_dir = Path("data/magichub/asr-sfdusc/WAV")

raw_data = raw_data[raw_data["PROMPT"].notna()]


epi = Backoff(["tgl-Latn", "eng-Latn"])

data = pd.DataFrame(
    {
        "audio_filepath": [base_dir / path for path in raw_data["path"]],
        "text": [epi.transliterate(text).strip() for text in raw_data["TRANSCRIPTION"]],
    }
)

data["duration"] = [get_wav_duration(filename) for filename in data["audio_filepath"]]

print(data)

train_test_val_split(data, "data/magichub/asr-sfdusc/")
