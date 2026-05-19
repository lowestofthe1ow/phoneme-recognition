import json

import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from datasets import DatasetDict

RANDOM_STATE = 765
# MAX_AUDIO_LEN = 160000  # 10s at 16kHz
TARGET_SR = 16000  # TODO: Look into sampling rates


class PhonemeDataset(Dataset):
    def __init__(self, manifest_path, tokenizer):
        with open(manifest_path) as f:
            self.samples = [json.loads(line) for line in f]

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        waveform, sr = torchaudio.load(sample["audio_filepath"])

        # Resample to keep sampling rate consistent
        if sr != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

        return {
            "audio_values": waveform.mean(0),  # [:MAX_AUDIO_LEN],
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
