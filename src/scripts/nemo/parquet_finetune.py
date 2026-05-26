import io
import json
import os
import time
from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import soundfile as sf
import torch
from nemo.collections.asr.models import ASRModel
from nemo.core.config import hydra_runner
from nemo.utils import logging, model_utils
from nemo.utils.exp_manager import exp_manager
from nemo.utils.get_rank import is_global_rank_zero
from nemo.utils.trainer_utils import resolve_trainer_cfg
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class ParquetAudioDataset(Dataset):
    def __init__(
        self,
        manifest_filepath,
        base_dir="data/filipinospeechcorpus/data",
        min_duration=0.10,
        max_duration=45.0,
    ):
        self.samples = []
        self.base_dir = Path(base_dir)

        skipped_too_long = 0
        skipped_too_short = 0

        with open(manifest_filepath, "r") as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    duration = sample.get("duration", 0)

                    if duration < min_duration:
                        skipped_too_short += 1
                    elif duration > max_duration:
                        skipped_too_long += 1
                    else:
                        self.samples.append(sample)

        print(f"Loaded {len(self.samples)} valid samples from {manifest_filepath}.")
        print(
            f"Skipped {skipped_too_short} samples because they were under {min_duration}s."
        )
        print(
            f"Skipped {skipped_too_long} samples because they exceeded the {max_duration}s limit."
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

        if len(arr) < 4000:
            import numpy as np

            padding_needed = 4000 - len(arr)
            # Add padding zeros (silence) to the end of the clip
            arr = np.pad(arr, (0, padding_needed), mode="constant")
        # --------------------------------------------

        return {
            "audio_signal": torch.tensor(arr, dtype=torch.float32),
            "audio_len": torch.tensor(len(arr), dtype=torch.int32),
            "transcript": text,
            "audio_filepath": virtual_path,
        }


def get_base_model(trainer, cfg):
    nemo_model_path = cfg.get("init_from_nemo_model", None)
    pretrained_name = cfg.get("init_from_pretrained_model", None)

    if nemo_model_path and pretrained_name:
        raise ValueError(
            "Pass either init_from_nemo_model or init_from_pretrained_model, not both."
        )
    elif not nemo_model_path and not pretrained_name:
        raise ValueError("Must provide at least one base initialization model.")

    if nemo_model_path:
        asr_model = ASRModel.restore_from(restore_path=nemo_model_path)
    else:
        asr_model = ASRModel.from_pretrained(model_name=pretrained_name)

    asr_model.set_trainer(trainer)
    return asr_model


def check_vocabulary(asr_model, cfg):
    if (
        hasattr(cfg.model.tokenizer, "update_tokenizer")
        and cfg.model.tokenizer.update_tokenizer
    ):
        vocab_size = asr_model.tokenizer.vocab_size
        decoder = asr_model.decoder.state_dict()
        joint_state = (
            asr_model.joint.state_dict() if hasattr(asr_model, "joint") else None
        )

        logging.info("Updating tokenizer via script specifications...")
        asr_model.change_vocabulary(
            new_tokenizer_dir=cfg.model.tokenizer.dir,
            new_tokenizer_type=cfg.model.tokenizer.type,
        )
        if asr_model.tokenizer.vocab_size == vocab_size:
            asr_model.decoder.load_state_dict(decoder)
            if joint_state is not None:
                asr_model.joint.load_state_dict(joint_state)
    return asr_model


def manual_bpe_collate_fn(batch, tokenizer):
    """
    Manually replicates NeMo's BPE data collation.
    Takes a batch of dicts from ParquetAudioDataset and processes them into
    padded tensors that EncDecCTCModelBPE expects.
    """
    audio_signals = []
    audio_lengths = []
    tokens_list = []
    tokens_lengths = []

    for sample in batch:
        sig = sample["audio_signal"]
        audio_signals.append(sig)
        audio_lengths.append(sample["audio_len"])

        token_ids = tokenizer.text_to_ids(sample["transcript"])
        token_tensor = torch.tensor(token_ids, dtype=torch.long)

        tokens_list.append(token_tensor)
        tokens_lengths.append(torch.tensor(len(token_ids), dtype=torch.long))

    padded_audio = pad_sequence(audio_signals, batch_first=True, padding_value=0.0)
    padded_tokens = pad_sequence(
        tokens_list, batch_first=True, padding_value=-1
    )  # NeMo defaults to -1 or 0 for blank padding

    audio_lengths = torch.stack(audio_lengths)
    tokens_lengths = torch.stack(tokens_lengths)

    return padded_audio, audio_lengths, padded_tokens, tokens_lengths


def setup_dataloaders(asr_model, cfg):
    cfg = model_utils.convert_model_config_to_dict_config(cfg)
    logging.info(
        "Injecting custom in-memory Parquet dataset loops with a manual collator..."
    )

    collate_wrapper = lambda b: manual_bpe_collate_fn(b, tokenizer=asr_model.tokenizer)

    train_manifest = cfg.model.train_ds.manifest_filepath
    train_dataset = ParquetAudioDataset(train_manifest)

    asr_model._train_dl = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=4,
        shuffle=cfg.model.train_ds.get("shuffle", True),
        num_workers=0,  # Keep at 0 for in-memory parquet cache safety
        collate_fn=collate_wrapper,
    )

    val_manifest = cfg.model.validation_ds.manifest_filepath
    val_dataset = ParquetAudioDataset(val_manifest)

    asr_model._validation_dl = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_wrapper,
    )

    return asr_model


@hydra_runner(config_path="conf/asr_finetune", config_name="speech_to_text_finetune")
def main(cfg):
    logging.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")

    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))

    asr_model = get_base_model(trainer, cfg)
    asr_model = check_vocabulary(asr_model, cfg)
    asr_model = setup_dataloaders(asr_model, cfg)

    asr_model.setup_optimization(cfg.model.optim)

    if hasattr(cfg.model, "spec_augment") and cfg.model.spec_augment is not None:
        asr_model.spec_augment = ASRModel.from_config_dict(cfg.model.spec_augment)

    trainer.fit(asr_model)


if __name__ == "__main__":
    main()
