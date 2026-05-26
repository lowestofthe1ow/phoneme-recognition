# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io

# Copyright (c) 2026, Custom Parquet Adaptation.
import json
import os
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Optional

import jiwer
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.utils.transcribe_utils import (
    PunctuationCapitalization,
    TextProcessingConfig,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf import MISSING, OmegaConf, open_dict
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ParquetAudioDataset(Dataset):
    def __init__(
        self,
        manifest_filepath,
        base_dir="data/filipinospeechcorpus/data",
        min_duration=0.10,
        max_duration=75.0,
    ):
        self.samples = []
        self.base_dir = Path(base_dir)

        with open(manifest_filepath, "r") as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    duration = sample.get("duration", 0)
                    if min_duration <= duration <= max_duration:
                        self.samples.append(sample)

        logging.info(
            f"[{Path(manifest_filepath).name}] Loaded {len(self.samples)} valid samples for evaluation."
        )
        self.parquet_cache = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        virtual_path = sample["audio_filepath"]
        text = sample["text"]

        parts = virtual_path.replace("fsc://", "").split("/")
        parquet_name = parts[0]
        row_idx = int(parts[1].split("_")[1].split(".")[0])

        if parquet_name not in self.parquet_cache:
            self.parquet_cache[parquet_name] = pd.read_parquet(
                self.base_dir / parquet_name
            )

        df = self.parquet_cache[parquet_name]
        audio_bytes = df.iloc[row_idx]["audio"]["bytes"]

        arr, sr = sf.read(io.BytesIO(audio_bytes))

        # Guardrail for ultra-short utterances (below 0.25 seconds / 4000 samples at 16kHz)
        if len(arr) < 4000:
            padding_needed = 4000 - len(arr)
            arr = np.pad(arr, (0, padding_needed), mode="constant")

        return {
            "audio_signal": torch.tensor(arr, dtype=torch.float32),
            "audio_len": torch.tensor(len(arr), dtype=torch.int32),
            "transcript": text,
            "audio_filepath": virtual_path,
        }


def manual_eval_collate_fn(batch):
    audio_signals = [item["audio_signal"] for item in batch]
    audio_lens = [item["audio_len"] for item in batch]
    transcripts = [item["transcript"] for item in batch]
    audio_filepaths = [item["audio_filepath"] for item in batch]

    # Pad audio sequences dynamically within this batch
    padded_signals = torch.nn.utils.rnn.pad_sequence(
        audio_signals, batch_first=True, padding_value=0.0
    )

    return {
        "audio_signal": padded_signals,
        "audio_len": torch.stack(audio_lens),
        "transcripts": transcripts,
        "audio_filepaths": audio_filepaths,
    }


@dataclass
class EvaluationConfig:
    model_path: Optional[str] = None
    pretrained_name: Optional[str] = None
    dataset_manifest: str = MISSING
    output_filename: Optional[str] = "evaluation_transcripts.json"
    batch_size: int = 16
    amp: bool = True

    gt_text_attr_name: str = "text"
    use_cer: bool = False
    tolerance: Optional[float] = None

    text_processing: Optional[TextProcessingConfig] = field(
        default_factory=lambda: TextProcessingConfig(
            punctuation_marks=".,?",
            separate_punctuation=False,
            do_lowercase=False,
            rm_punctuation=False,
        )
    )


@hydra_runner(config_name="EvaluationConfig", schema=EvaluationConfig)
def main(cfg: EvaluationConfig):
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if not os.path.exists(cfg.dataset_manifest):
        raise FileNotFoundError(
            f"The dataset manifest file could not be found at path: {cfg.dataset_manifest}"
        )

    # Load your fine-tuned .nemo model checkpoint
    logging.info(f"Loading ASR model from checkpoint: {cfg.model_path}")
    if cfg.model_path.endswith(".nemo"):
        asr_model = ASRModel.restore_from(cfg.model_path, map_location=device)
    else:
        raise ValueError(
            "Please provide a valid path to your fine-tuned '.nemo' file checkpoint."
        )

    asr_model.eval()

    # Initialize the custom Parquet evaluation Dataset
    eval_dataset = ParquetAudioDataset(
        manifest_filepath=cfg.dataset_manifest, min_duration=0.13, max_duration=75.0
    )
    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,  # Keeps file operations stable in-memory
        collate_fn=manual_eval_collate_fn,
    )

    ground_truth_text = []
    predicted_text = []
    output_records = []

    logging.info("Starting transcription processing over Parquet binary blobs...")

    autocast_context = torch.cuda.amp.autocast(enabled=cfg.amp)

    with autocast_context:
        for batch in tqdm(eval_loader, desc="Evaluating Batches"):
            signals = batch["audio_signal"].to(device)
            signal_lens = batch["audio_len"].to(device)

            log_probs, encoded_len, predictions = asr_model.forward(
                input_signal=signals, input_signal_length=signal_lens
            )

            hypotheses = asr_model.decoding.ctc_decoder_predictions_tensor(
                predictions, decoder_lengths=encoded_len
            )

            for b_idx in range(len(batch["transcripts"])):
                gt = str(batch["transcripts"][b_idx]).strip()
                pred = hypotheses[b_idx]

                if isinstance(pred, list) or not isinstance(pred, str):
                    pred = getattr(pred, "text", str(pred))
                pred = pred.strip()

                ground_truth_text.append(gt)
                predicted_text.append(pred)

                # --- CALCULATE PER-SAMPLE CER ---
                if gt == pred:
                    sample_cer = 0.0
                elif len(gt) == 0:
                    sample_cer = 1.0 if len(pred) > 0 else 0.0
                else:
                    sample_cer = jiwer.cer(gt, pred)
                # ---------------------------------

                # Append sample data, now incorporating the individual CER score
                output_records.append(
                    {
                        "audio_filepath": batch["audio_filepaths"][b_idx],
                        "text": gt,
                        "pred_text": pred,
                        "cer": float(sample_cer),  # Added metric payload
                    }
                )
    # Apply Text Post-Processing Normalizations matching standard scripts
    pc = PunctuationCapitalization(cfg.text_processing.punctuation_marks)
    if cfg.text_processing.separate_punctuation:
        ground_truth_text = pc.separate_punctuation(ground_truth_text)
        predicted_text = pc.separate_punctuation(predicted_text)
    if cfg.text_processing.do_lowercase:
        ground_truth_text = pc.do_lowercase(ground_truth_text)
        predicted_text = pc.do_lowercase(predicted_text)
    if cfg.text_processing.rm_punctuation:
        ground_truth_text = pc.rm_punctuation(ground_truth_text)
        predicted_text = pc.rm_punctuation(predicted_text)

    # Compute Final Error Metrics
    cer = word_error_rate(
        hypotheses=predicted_text, references=ground_truth_text, use_cer=True
    )
    wer = word_error_rate(
        hypotheses=predicted_text, references=ground_truth_text, use_cer=False
    )

    metric_name = "CER" if cfg.use_cer else "WER"
    metric_value = cer if cfg.use_cer else wer

    logging.info(f"==== FINAL PERFORMANCE SCORES ====")
    logging.info(f"Dataset Word Error Rate (WER): {wer:.2%}")
    logging.info(f"Dataset Character Error Rate (CER): {cer:.2%}")
    logging.info(f"==================================")

    if cfg.tolerance is not None and metric_value > cfg.tolerance:
        raise ValueError(
            f"Got {metric_name} of {metric_value:.2%}, which exceeded tolerance={cfg.tolerance:.2%}"
        )

    # Write transcript logs to output file
    if cfg.output_filename:
        with open(cfg.output_filename, "w") as out_f:
            for item in output_records:
                out_f.write(json.dumps(item) + "\n")
        logging.info(
            f"Saved evaluation text transcripts output manifest to: {cfg.output_filename}"
        )

    # Return Hydra parameters dictionary structure
    with open_dict(cfg):
        cfg.metric_name = metric_name
        cfg.metric_value = metric_value

    return cfg


if __name__ == "__main__":
    main()
