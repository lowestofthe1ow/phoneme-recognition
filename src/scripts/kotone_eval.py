import editdistance
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from src.datasets.pr_dataset import PhonemeDataCollator, dataset_from_manifests
from src.engines.kotone import build_model

TRAIN_MANIFEST = "data/nexdata/filipino_822/train_manifest.json"
VALID_MANIFEST = "data/nexdata/filipino_822/valid_manifest.json"
TEST_MANIFEST = "data/nexdata/filipino_822/test_manifest.json"

DEFAULT_MODEL_ID = "charsiu/g2p_multilingual_byT5_small_100"

CHECKPOINT = "models/checkpoints/2026-05-30_15-19_kotone/checkpoint-1014"

BATCH_SIZE = 8

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_ID)

model = build_model()
model.load_state_dict(load_file(f"{CHECKPOINT}/model.safetensors"), strict=False)
model.to(device)
model.eval()

dataset = dataset_from_manifests(
    TRAIN_MANIFEST, VALID_MANIFEST, TEST_MANIFEST, tokenizer
)
data_collator = PhonemeDataCollator(tokenizer)
test_set = dataset["test"]

total_errors, total_chars = 0, 0

for i in range(0, len(test_set), BATCH_SIZE):
    batch_list = [
        {k: v for k, v in test_set[j].items() if k in ["audio_values", "labels"]}
        for j in range(i, min(i + BATCH_SIZE, len(test_set)))
    ]

    inputs = data_collator(batch_list)
    audio = inputs["audio_values"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        generated = model.generate(audio, attention_mask, max_length=256)

    labels = inputs["labels"].clone()
    labels[labels == -100] = tokenizer.pad_token_id

    preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
    refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Calculate (POOLED) CER as well as print predictions
    # TODO: PER and PFER
    for p, l in zip(preds, refs):
        total_errors += editdistance.eval(p, r)
        total_chars += len(r)

        print("-" * 40)
        print(f"Target:  {l}")
        print(f"Predict: {p}")
        print("=" * 40)

print(f"CER: {total_errors / total_chars:.4f} ({total_errors}/{total_chars})")
