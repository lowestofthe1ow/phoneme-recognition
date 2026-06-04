import gc
import os
from pathlib import Path

import librosa
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

from src.utils.sliding_window import predict_sliding_window_batch

RANDOM_STATE = 339
CHECKPOINT = "models/g2p/checkpoint-9670"
BATCH_SIZE = 8
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading G2P Model on {device}...")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
model = T5ForConditionalGeneration.from_pretrained(CHECKPOINT).to(device)


def get_wav_duration(path):
    return librosa.get_duration(path=str(path))


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
    print(
        f"[{type}] Filtered out {initial_count - len(raw_data)} rows containing numbers."
    )

    wav_files = [
        str(tsv_directory / "audio" / type / filename) for filename in raw_data["audio"]
    ]

    raw_data["sentence"] = (
        raw_data["sentence"].astype(str).str.replace(r"[()]", "", regex=True)
    )
    texts = raw_data["sentence"].astype(str).tolist()
    predictions = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"G2P {type}"):
        batch_sentences = texts[i : i + BATCH_SIZE]

        batch_preds = predict_sliding_window_batch(
            sentences=batch_sentences,
            model=model,
            tokenizer=tokenizer,
            window_size=11,
            max_batch_size=32,  # Safe inner-batch size for the GPU
        )
        predictions.extend(batch_preds)

    data = pd.DataFrame(
        {
            "audio_filepath": wav_files,
            "text": predictions,
            "duration": [
                get_wav_duration(f) for f in tqdm(wav_files, desc=f"Durations {type}")
            ],
        }
    )

    data["text"] = (
        data["text"]
        .str.replace(",", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace(r"-(?!ʔ)", "ʔ", regex=True)
        .str.replace("-", "", regex=False)
    )

    manifest_name = "valid" if type == "dev" else type
    output_path = Path(tsv_directory) / f"{manifest_name}_manifest.json"

    data.to_json(
        output_path,
        orient="records",
        lines=True,
        default_handler=str,
    )
    print(f"Saved {len(data)} entries to {output_path}\n")


if __name__ == "__main__":
    for split in ["train", "test", "dev"]:
        make_manifest(split)
        torch.cuda.empty_cache()
        gc.collect()
