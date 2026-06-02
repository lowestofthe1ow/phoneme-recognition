import json
import os
import re

import evaluate
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

# --- Configuration ---
CHECKPOINT_PATH = "models/exp/mt5_p2g_fil/checkpoint-1316"
EVAL_DATA_PATH = "data/evaluation_transcripts.json"
TEST_TARGETS_PATH = "data/google/fleurs/fil_ph/test_aligned.json"

MAX_INPUT_LENGTH = 160
BATCH_SIZE = 16


def clean_grapheme(text):
    """Keep this identical to your training script for fair evaluation."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9ñ\-\'\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    print(f"### Loading Tokenizer and Model from {CHECKPOINT_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_bf16_supported():
        model = model.to(torch.bfloat16)
    model = model.to(device)
    model.eval()

    gen_config = GenerationConfig.from_model_config(model.config)
    gen_config.decoder_start_token_id = model.config.pad_token_id
    gen_config.bos_token_id = model.config.pad_token_id
    gen_config.eos_token_id = tokenizer.eos_token_id
    gen_config.pad_token_id = tokenizer.pad_token_id
    gen_config.max_length = 120

    gen_config.num_beams = 4
    gen_config.repetition_penalty = 1.2
    gen_config.no_repeat_ngram_size = 3
    gen_config.early_stopping = True

    model.generation_config = gen_config

    print("### Loading Datasets...")
    input_phonemes = []
    with open(EVAL_DATA_PATH, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith("["):
            eval_data = json.loads(content)
            input_phonemes = [item["pred_text"].replace("'", "ˈ") for item in eval_data]
        else:
            for line in content.split("\n"):
                if line.strip():
                    item = json.loads(line)
                    input_phonemes.append(item["pred_text"].replace("'", "ˈ"))

    targets = []
    with open(TEST_TARGETS_PATH, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith("["):
            test_data = json.loads(content)
        else:
            test_data = [
                json.loads(line) for line in content.split("\n") if line.strip()
            ]

        for item in test_data:
            targets.append(clean_grapheme(item["grapheme"]))

    assert len(input_phonemes) == len(
        targets
    ), f"Data mismatch! Found {len(input_phonemes)} inputs but {len(targets)} targets."

    print(f"### Starting Batch Inference on {len(input_phonemes)} samples...")
    predictions = []

    for i in tqdm(range(0, len(input_phonemes), BATCH_SIZE)):
        batch = input_phonemes[i : i + BATCH_SIZE]
        inputs = [f"p2g: {p}" for p in batch]

        encoded = tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(**encoded)

        decoded_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for pred in decoded_batch:
            clean_pred = re.sub(r"<extra_id_\d+>", "", pred).strip()
            predictions.append(clean_pred)

    print("### Calculating Overall Metrics...")
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    final_wer = wer_metric.compute(predictions=predictions, references=targets)
    final_cer = cer_metric.compute(predictions=predictions, references=targets)

    print("\n" + "=" * 50)
    print(f"FINAL TEST WER: {final_wer:.4f} ({final_wer * 100:.2f}%)")
    print(f"FINAL TEST CER: {final_cer:.4f} ({final_cer * 100:.2f}%)")
    print("=" * 50 + "\n")

    # --- Error Analysis Section ---
    print("### Analyzing best and worst predictions...")

    # Calculate unique WER and CER scores for each separate sentence pair
    sample_wers = []
    sample_cers = []
    for pred, ref in zip(predictions, targets):
        if not ref.strip() and not pred.strip():
            wer_score = 0.0
            cer_score = 0.0
        elif not ref.strip():
            wer_score = 1.0
            cer_score = 1.0
        else:
            wer_score = wer_metric.compute(predictions=[pred], references=[ref])
            cer_score = cer_metric.compute(predictions=[pred], references=[ref])
        sample_wers.append(wer_score)
        sample_cers.append(cer_score)

    sample_wers = np.array(sample_wers)
    sample_cers = np.array(sample_cers)
    sorted_indices = np.argsort(
        sample_wers
    )  # Sorting prioritized by word collapse (WER)

    def print_examples(indices, label_title):
        print("\n" + "#" * 70)
        print(f"  TOP 5 {label_title} PREDICTIONS")
        print("#" * 70)
        for rank, idx in enumerate(indices):
            print(
                f"[{rank + 1}] (Index: {idx}) | Sample WER: {sample_wers[idx]:.4f} | Sample CER: {sample_cers[idx]:.4f}"
            )
            print(f"  Phoneme Input : {input_phonemes[idx]}")
            print(f"  True Grapheme : {targets[idx]}")
            print(f"  Model Output  : {predictions[idx]}")
            print("-" * 60)

    # Top 5 Best: lowest WER scores
    best_indices = sorted_indices[:5]
    print_examples(best_indices, "BEST")

    # Top 5 Worst: highest WER scores (reversed sort order)
    worst_indices = sorted_indices[::-1][:5]
    print_examples(worst_indices, "WORST")


if __name__ == "__main__":
    main()
