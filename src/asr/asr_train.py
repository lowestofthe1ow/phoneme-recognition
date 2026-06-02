import json
import os
import re

import evaluate
import numpy as np
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from datasets import Dataset, DatasetDict

MODEL_NAME = "google/mt5-base"
DATA_DIR = "data/google/fleurs/fil_ph"
OUTPUT_DIR = "models/exp/mt5_p2g_fil"

TRAIN_PATH = os.path.join(DATA_DIR, "train_aligned.json")
VALID_PATH = os.path.join(DATA_DIR, "valid_aligned.json")
TEST_PATH = os.path.join(DATA_DIR, "test_aligned.json")

MAX_INPUT_LENGTH = 160
MAX_TARGET_LENGTH = 120

wer_metric = evaluate.load("wer")


def clean_grapheme(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9ñ\-\'\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_local_manifest(file_path):
    entries = []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            raise ValueError(f"File {file_path} is empty.")

        if content.startswith("["):
            entries = json.loads(content)
        else:
            f.seek(0)
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))

    inputs = []
    targets = []
    for entry in entries:
        phoneme = entry["phoneme"]

        phoneme = phoneme.replace("'", "ˈ")

        grapheme = clean_grapheme(entry["grapheme"])

        inputs.append(f"p2g: {phoneme}")
        targets.append(grapheme)

    return Dataset.from_dict({"input_text": inputs, "target_text": targets})


def main():
    print("### Loading and cleaning datasets...")
    raw_datasets = DatasetDict(
        {
            "train": load_local_manifest(TRAIN_PATH),
            "validation": load_local_manifest(VALID_PATH),
            "test": load_local_manifest(TEST_PATH),
        }
    )
    print(f"Loaded {len(raw_datasets['train'])} training samples.")

    print(f"### Initializing tokenizer and model from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    gen_config = GenerationConfig.from_model_config(model.config)
    gen_config.decoder_start_token_id = model.config.pad_token_id
    gen_config.bos_token_id = model.config.pad_token_id
    gen_config.eos_token_id = tokenizer.eos_token_id
    gen_config.pad_token_id = tokenizer.pad_token_id
    gen_config.max_length = MAX_TARGET_LENGTH

    gen_config.num_beams = 4
    gen_config.repetition_penalty = 1.2
    gen_config.no_repeat_ngram_size = 3
    gen_config.early_stopping = True

    model.generation_config = gen_config

    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["input_text"], max_length=MAX_INPUT_LENGTH, truncation=True
        )
        labels = tokenizer(
            text_target=examples["target_text"],
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("### Tokenizing datasets...")
    tokenized_datasets = raw_datasets.map(
        tokenize_function, batched=True, remove_columns=["input_text", "target_text"]
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Replace -100 masking with pad tokens
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # Decode BOTH predictions and the perfectly-aligned labels
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Scrub out <extra_id_0> tokens and dangling spaces
        decoded_preds = [
            re.sub(r"<extra_id_\d+>", "", pred).strip() for pred in decoded_preds
        ]
        decoded_labels = [label.strip() for label in decoded_labels]

        print("\n" + "=" * 70)
        print("   VAL GENERATION EXAMPLES (Phoneme -> Grapheme Prediction)   ")
        print("=" * 70)
        num_to_print = min(5, len(decoded_preds))
        for i in range(num_to_print):
            print(f"Example {i+1}:")
            print(f"  Expected Target  : {decoded_labels[i]}")
            print(f"  Model Prediction : {decoded_preds[i]}")
            print("-" * 50)
        print("=" * 70 + "\n")

        wer = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"wer": wer}

    print("### Setting up training arguments...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        optim="adafactor",
        weight_decay=0.01,
        label_smoothing_factor=0.1,
        save_total_limit=2,
        num_train_epochs=200,
        predict_with_generate=True,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=False,
        logging_steps=100,
        report_to="none",
        metric_for_best_model="wer",
        greater_is_better=False,
        load_best_model_at_end=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("### Starting fine-tuning...")
    trainer.train()

    print(f"### Fine-tuning complete. Saving best model checkpoint to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
