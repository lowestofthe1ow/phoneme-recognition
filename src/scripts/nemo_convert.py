# Use this to convert .nemo models
# Copy phoneme_recognition.nemo into "models/nvidia/" directory
# Copy ipa_tokenizer folder into "models/tokenizers/" directory

import os
import tarfile

import nemo.collections.asr as nemo_asr
import torch
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    SequenceFeatureExtractor,
    T5Tokenizer,
)


## 1. Define the Custom Hugging Face Wrapper Classes
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

        # FIX: Explicitly map to CPU to prevent meta-tensor inheritance
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

        # Process forward pass using the underlying NeMo layers
        log_probs, encoded_len, _ = self.nemo_model(
            input_signal=input_values, input_signal_length=input_lengths
        )

        # Return format expected by Hugging Face Trainer and Pipelines
        from transformers.modeling_outputs import TokenClassifierOutput

        return TokenClassifierOutput(loss=None, logits=log_probs)


## 2. Main Execution Block
def main():
    nemo_path = "models/nvidia/phoneme_recognition.nemo"
    output_dir = "models/nvidia/hf_phoneme_recognition_model"

    print(f"[*] Creating target directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Extract Tokenizer from the .nemo tarball
    print(f"[*] Extracting tokenizer file from {nemo_path}...")
    temp_tokenizer_path = os.path.join(output_dir, "extracted_tokenizer.model")

    with tarfile.open(nemo_path, "r:*") as tar:
        tokenizer_extracted = False
        for member in tar.getmembers():
            if member.name.endswith(".model"):
                print(f"[+] Found tokenizer file inside archive: {member.name}")
                # Flatten the extraction path to avoid subfolder clutter
                member.name = os.path.basename(member.name)
                tar.extract(member, path=output_dir)

                extracted_file_path = os.path.join(output_dir, member.name)
                if os.path.exists(temp_tokenizer_path):
                    os.remove(temp_tokenizer_path)
                os.rename(extracted_file_path, temp_tokenizer_path)
                tokenizer_extracted = True
                break

        if not tokenizer_extracted:
            raise FileNotFoundError(
                "Could not locate a SentencePiece '.model' file inside the .nemo archive."
            )

    # FIX: Instantiate the model directly for initial conversion (Do not use from_pretrained here)
    print(f"[*] Initializing model configuration from: {nemo_path}")
    config = NeMoConformerConfig(nemo_model_path=nemo_path)

    print("[*] Wrapping NeMo model weights into Hugging Face structure...")
    model = NeMoConformerForCTC(config)

    # Save Model Weights and Config JSON
    print(f"[*] Saving Hugging Face model weights and config to {output_dir}...")
    model.save_pretrained(output_dir)

    # Package the Tokenizer into standard HF format
    print("[*] Generating and saving Hugging Face tokenizer configuration...")
    tokenizer = T5Tokenizer(vocab_file=temp_tokenizer_path)
    tokenizer.save_pretrained(output_dir)

    # Clean up the loose extracted binary model file
    if os.path.exists(temp_tokenizer_path):
        os.remove(temp_tokenizer_path)

    # Package the Feature Extractor configuration
    print(
        "[*] Generating and saving Hugging Face audio feature extractor configuration..."
    )
    feature_extractor = SequenceFeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0
    )
    feature_extractor.save_pretrained(output_dir)

    print(
        f"\n[SUCCESS] Conversion complete! The model in '{output_dir}' is now fully ready for the Hugging Face Trainer."
    )


if __name__ == "__main__":
    main()
