"""
This script creates three files, test_manifest.json, train_manifest.json, and
valid_manifest.json, in the data/filipinospeechcorpus/data directory.
"""

import io
import os
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

CHECKPOINT = "models/g2p/checkpoint-9670"
BATCH_SIZE = 8
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
model = T5ForConditionalGeneration.from_pretrained(CHECKPOINT).to(device)


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


def process_parquet_dataset(directory_path):
    """Reads parquet files, skips audio extraction, runs G2P, and aggregates data."""
    directory = Path(directory_path)

    if not directory.exists():
        raise FileNotFoundError(f"The directory {directory.resolve()} does not exist!")

    parquet_files = [
        directory / f for f in os.listdir(directory) if f.endswith(".parquet")
    ]
    print(f"Found {len(parquet_files)} parquet files.")

    all_audio_paths = []
    all_predictions = []
    all_durations = []

    for parquet_file in parquet_files:
        print(f"\nProcessing {parquet_file.name}...")
        df = pd.read_parquet(parquet_file)

        local_row_idx = 0

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing metadata"):
            duration = row["duration"]

            # Map the exact local DataFrame row index into the URI
            virtual_filename = (
                f"fsc://{parquet_file.name}/sample_{local_row_idx:06d}.wav"
            )

            all_audio_paths.append(virtual_filename)
            all_durations.append(duration)
            local_row_idx += 1

        texts = (
            df["sentence"]
            .astype(str)
            .str.strip()
            .apply(
                lambda s: (
                    s if s.endswith((".", "!", "?")) or len(s.split()) == 1 else s + "."
                )
            )
            .str.capitalize()
            .tolist()
        )

        for i in tqdm(
            range(0, len(texts), BATCH_SIZE), desc=f"G2P {parquet_file.name}"
        ):
            batch = texts[i : i + BATCH_SIZE]
            all_predictions.extend(predict_batch(batch))

    data = pd.DataFrame(
        {
            "audio_filepath": all_audio_paths,
            "text": all_predictions,
            "duration": all_durations,
        }
    )

    data["text"] = (
        data["text"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(r"-(?!ʔ)", "ʔ", regex=True)
        .str.replace("-", "", regex=False)
    )

    return data


BASE_DIR = "data/filipinospeechcorpus/data"
data = process_parquet_dataset(BASE_DIR)

if not data.empty:
    print("\nSplitting dataset into 80/10/10 (Train/Test/Dev)...")

    train_dev_df, test_df = train_test_split(data, test_size=0.10, random_state=42)
    train_df, dev_df = train_test_split(
        train_dev_df, test_size=0.11111, random_state=42
    )

    output_path = Path(BASE_DIR)
    train_df.to_json(output_path / "train_manifest.json", orient="records", lines=True)
    test_df.to_json(output_path / "test_manifest.json", orient="records", lines=True)
    dev_df.to_json(output_path / "valid_manifest.json", orient="records", lines=True)
