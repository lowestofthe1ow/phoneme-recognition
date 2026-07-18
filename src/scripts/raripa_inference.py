import argparse

import torch
import torchaudio
from safetensors.torch import load_file
from transformers import AutoTokenizer

from src.engines.raripa_build import build_model

DEFAULT_MODEL_ID = "charsiu/g2p_multilingual_byT5_small_100"
TARGET_SR = 16000

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint-path", required=True)
parser.add_argument("--wav-path", required=True)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_ID)
model = build_model()
model.load_state_dict(
    load_file(f"{args.checkpoint_path}/model.safetensors"), strict=False
)
model.to(device)
model.eval()

waveform, sr = torchaudio.load(args.wav_path)
if sr != TARGET_SR:
    waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
waveform = waveform.mean(0)
waveform = (waveform - waveform.mean()) / torch.sqrt(waveform.var() + 1e-7)

audio = waveform.unsqueeze(0).to(device)  # (1, T)
attention_mask = torch.ones_like(audio, dtype=torch.long)  # no padding, all 1s

with torch.no_grad():
    generated = model.generate(audio, attention_mask)

print(tokenizer.decode(generated[0], skip_special_tokens=True))
