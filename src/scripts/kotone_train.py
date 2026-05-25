import argparse
import json
from datetime import datetime

import torch
from transformers import (
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

from src.datasets.pr_dataset import PhonemeDataCollator, dataset_from_manifests
from src.engines.kotone import build_model
from src.utils.preview_callback import PreviewCallback

# Requires manifest files following NeMo's format
TRAIN_MANIFEST = "data/nexdata/filipino_822/train_manifest.json"
VALID_MANIFEST = "data/nexdata/filipino_822/valid_manifest.json"
TEST_MANIFEST = "data/nexdata/filipino_822/test_manifest.json"

DEFAULT_MODEL_ID = "charsiu/g2p_multilingual_byT5_small_100"
MAX_AUDIO_LEN = 160000

parser = argparse.ArgumentParser()
parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
parser.add_argument("--train-manifest-path", default=TRAIN_MANIFEST)
parser.add_argument("--valid-manifest-path", default=VALID_MANIFEST)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_ID)


model = build_model(tokenizer)
dataset = dataset_from_manifests(
    TRAIN_MANIFEST, VALID_MANIFEST, TEST_MANIFEST, tokenizer
)
data_collator = PhonemeDataCollator(tokenizer)

run_id = datetime.now().strftime("%Y-%m-%d_%H-%M")

training_args = Seq2SeqTrainingArguments(
    output_dir=f"./models/checkpoints/{run_id}_kotone",
    bf16=True,
    # fp16=True,
    optim="adafactor",
    # --------------------------------------------
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    # --------------------------------------------
    # TODO: Look into NAS?
    learning_rate=3e-4,
    # lr_scheduler_type=...
    # warmup_steps=...
    # --------------------------------------------
    num_train_epochs=200,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    logging_steps=10,
    # --------------------------------------------
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    remove_unused_columns=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    callbacks=[PreviewCallback(dataset["validation"], data_collator, tokenizer)],
)

trainer.train()
