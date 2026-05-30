#!/usr/bin/env python3
import argparse
import json
import os

import librosa
import nemo.collections.asr as nemo_asr
import torch
from jiwer import cer, wer
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    SequenceFeatureExtractor,
    T5Tokenizer,
)
from transformers.modeling_outputs import TokenClassifierOutput


# 1. Re-declare Custom Wrapper Classes for HF Compatibility
class NeMoConformerConfig(PretrainedConfig):
    model_type = "nemo_conformer_ctc"

    def __init__(self, nemo_model_path=None, **kwargs):
        self.nemo_model_path = nemo_model_path
        super().__init__(**kwargs)


class NeMoConformerForCTC(PreTrainedModel):
    config_class = NeMoConformerConfig

    def __init__(self, config):
        super().__init__(config)
        if config.nemo_model_path is None:
            raise ValueError("nemo_model_path is missing from config.json.")

        # FIX APPLIED HERE: Map to CPU to prevent Hugging Face meta-tensor crashes
        self.nemo_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
            config.nemo_model_path, map_location="cpu"
        )
        self.nemo_model.preprocessor.featurizer.dither = 0.0

    def forward(self, input_values, attention_mask=None, labels=None):
        if attention_mask is not None:
            input_lengths = attention_mask.sum(dim=-1).long()
        else:
            input_lengths = torch.tensor(
                [input_values.shape[1]] * input_values.shape[0],
                device=input_values.device,
            )
        log_probs, encoded_len, _ = self.nemo_model(
            input_signal=input_values, input_signal_length=input_lengths
        )
        return TokenClassifierOutput(loss=None, logits=log_probs)


def load_manifest(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate converted Hugging Face Conformer-CTC models."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/nvidia/hf_phoneme_recognition_model",
        help="Path to HF model directory",
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default="models/tokenizers/ipa_tokenizer/tokenizer_spe_char_v32",
        help="Path to tokenizer model directory",
    )
    parser.add_argument(
        "--dataset_manifest",
        type=str,
        required=True,
        help="Path to NeMo JSON manifest file",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="evaluation_transcripts.json",
        help="Output file path",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Inference batch size"
    )
    parser.add_argument(
        "--use_cer", action="store_true", help="Fail based on CER instead of WER"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Maximum error rate threshold allowed",
    )
    parser.add_argument(
        "--scores_per_sample",
        action="store_true",
        help="Append metrics to each entry in the output manifest",
    )

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[*] Loading Model components from: {args.model_dir}")

    # 1. Load the config using YOUR custom class, NOT AutoConfig
    config = NeMoConformerConfig.from_pretrained(args.model_dir)

    # 2. Instantiate the model directly to bypass HF meta-contexts
    model = NeMoConformerForCTC(config)

    # 3. Move the model to the target device and set to eval mode
    model.to(device)
    model.eval()

    # Load packaged tokenizer and feature extractor directly via HF API
    tokenizer = SentencePieceProcessor()
    tokenizer.load(os.path.join(args.tokenizer_dir, "tokenizer.model"))

    print(f"[*] Parsing dataset manifest: {args.dataset_manifest}")
    manifest_entries = load_manifest(args.dataset_manifest)

    ground_truths = []
    predictions = []
    output_records = []

    print("[*] Launching batched audio inference...")
    with torch.no_grad():
        for i in tqdm(range(0, len(manifest_entries), args.batch_size)):
            batch_entries = manifest_entries[i : i + args.batch_size]
            audio_arrays = []

            for entry in batch_entries:
                wav, _ = librosa.load(entry["audio_filepath"], sr=16000)
                audio_arrays.append(wav)

            tensor_list = [
                torch.tensor(arr, dtype=torch.float32) for arr in audio_arrays
            ]

            # 2. Pad sequences to match the longest audio clip in the batch
            input_values = torch.nn.utils.rnn.pad_sequence(
                tensor_list, batch_first=True, padding_value=0.0
            ).to(device)

            # 3. Create the attention mask (1 for real audio, 0 for padding)
            seq_lengths = torch.tensor(
                [len(arr) for arr in audio_arrays], device=device
            )
            max_len = input_values.shape[1]
            attention_mask = torch.arange(max_len, device=device).expand(
                len(audio_arrays), max_len
            ) < seq_lengths.unsqueeze(1)
            attention_mask = attention_mask.long()

            outputs = model(input_values=input_values, attention_mask=attention_mask)
            logits = outputs.logits
            best_paths = torch.argmax(logits, dim=-1)

            for idx, entry in enumerate(batch_entries):
                raw_tokens = best_paths[idx].cpu().tolist()
                deduplicated_tokens = []
                prev_token = None
                blank_id = logits.shape[-1] - 1  # Standard CTC blank position for NeMo

                for token in raw_tokens:
                    if token != prev_token:
                        if token != blank_id:
                            deduplicated_tokens.append(token)
                    prev_token = token

                pred_text = tokenizer.decode_ids(deduplicated_tokens)
                gt_text = entry["text"]

                ground_truths.append(gt_text)
                predictions.append(pred_text)

                record = entry.copy()
                record["pred_text"] = pred_text

                if args.scores_per_sample:
                    record["sample_wer"] = (
                        round(wer(gt_text, pred_text), 4) if gt_text.strip() else 1.0
                    )
                    record["sample_cer"] = (
                        round(cer(gt_text, pred_text), 4) if gt_text.strip() else 1.0
                    )

                output_records.append(record)

    # Compute Global Dataset Metrics
    total_wer = wer(ground_truths, predictions)
    total_cer = cer(ground_truths, predictions)

    print("\n" + "=" * 50)
    print(" EVALUATION METRICS DASHBOARD")
    print("-" * 50)
    print(f" Global Dataset WER : {total_wer:.2%}")
    print(f" Global Dataset CER : {total_cer:.2%}")
    print("=" * 50 + "\n")

    print(f"[*] Writing transcript predictions manifest to: {args.output_filename}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_filename)), exist_ok=True)
    with open(args.output_filename, "w", encoding="utf-8") as out_f:
        for record in output_records:
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
