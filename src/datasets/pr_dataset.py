import io
import json

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from datasets import DatasetDict

RANDOM_STATE = 765
TARGET_SR = 16000
_parquet_cache: dict[str, pd.DataFrame] = {}


def _load_fsc(path: str) -> tuple[torch.Tensor, int]:
    parquet_file, sample_key = path.removeprefix("fsc://").rsplit("/", 1)
    if parquet_file not in _parquet_cache:
        _parquet_cache[parquet_file] = pd.read_parquet(
            "data/filipinospeechcorpus/data/" + parquet_file
        )
    idx = int(sample_key.removeprefix("sample_").removesuffix(".wav"))
    audio_bytes = _parquet_cache[parquet_file].loc[idx, "audio"]["bytes"]
    return torchaudio.load(io.BytesIO(audio_bytes))


class PhonemeDataset(Dataset):
    def __init__(self, manifest_path, tokenizer):
        import random  # Place at the top of your file, or here if needed

        with open(manifest_path) as f:
            self.samples = [json.loads(line) for line in f]

        # 1. Filter the samples first
        filtered_samples = [
            s
            for s in self.samples
            if 0.4 <= s["duration"] <= 10.0 and len(s["text"].split()) <= 10
        ]

        # 2. Add this line to take a random 10,000 sample subset (or less if the dataset is small)
        sampled_subset = random.sample(
            filtered_samples, min(len(filtered_samples), 10000)
        )

        # 3. Sort the final random subset
        self.samples = sorted(
            sampled_subset,
            key=lambda s: s["duration"],
        )

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        path = sample["audio_filepath"]
        load_fn = _load_fsc if path.startswith("fsc://") else torchaudio.load
        waveform, sr = load_fn(path)
        if sr != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        return {
            "audio_values": waveform.mean(0),
            "labels": self.tokenizer(
                sample["text"], return_tensors="pt"
            ).input_ids.squeeze(0),
        }


class PhonemeDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        audio = torch.nn.utils.rnn.pad_sequence(
            [item["audio_values"] for item in batch], batch_first=True
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [item["labels"] for item in batch], batch_first=True, padding_value=-100
        )
        decoder_input_ids = labels.clone()
        decoder_input_ids[decoder_input_ids == -100] = self.tokenizer.pad_token_id
        decoder_input_ids = torch.cat(
            [
                torch.full(
                    (decoder_input_ids.size(0), 1),
                    self.tokenizer.pad_token_id,
                    dtype=torch.long,
                ),
                decoder_input_ids[:, :-1],
            ],
            dim=1,
        )
        return {
            "audio_values": audio,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        }


def dataset_from_manifests(train_manifest, val_manifest, test_manifest, tokenizer):
    return DatasetDict(
        {
            "train": PhonemeDataset(train_manifest, tokenizer),
            "validation": PhonemeDataset(val_manifest, tokenizer),
            "test": PhonemeDataset(test_manifest, tokenizer),
        }
    )
