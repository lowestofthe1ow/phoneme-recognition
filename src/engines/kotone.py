import torch
import torch.nn as nn
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    T5ForConditionalGeneration,
    Wav2Vec2Model,
)

from src.engines.fusion import AudioProjection

WAV2VEC_HF = "Khalsuu/filipino-wav2vec2-l-xls-r-300m-official"
G2P_CHECKPOINT = "models/g2p/checkpoint-9670"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load wav2vec encoder
encoder = Wav2Vec2Model.from_pretrained(WAV2VEC_HF).to(device)

# Load ByT5 G2P decoder
g2p = T5ForConditionalGeneration.from_pretrained(
    G2P_CHECKPOINT,
)

decoder = g2p.decoder  # Decoder
lm_head = g2p.lm_head  # Linear head on top of the decoder

# Get rid of the rest of the G2P model to free up VRAM
del g2p
torch.cuda.empty_cache()

""" High-level overview of the architecture

┌───────────────────┐
│ ENCODER           │
│ filipino-wav2vec2 │
│ l-xls-r-300m      │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Downsampling +    │
│ Projection        │
└─────────┬─────────┘
          ├───────────┐
          ▼           │
┌───────────────────┐ │
│ DECODER           │ │
│ ByT5 from G2P     │ │
└────┬──────────────┘ │
     │                │
     ▼                ▼
┌──────────┐   ┌──────────┐
│ CE loss  ├─┬─┤ CTC loss │
└──────────┘ │ └──────────┘
             │
             ▼
┌─────────────────────────┐
│ Combined weighted loss  │
└─────────────────────────┘
"""


class KoToNe(PreTrainedModel):
    def __init__(self, config, encoder, decoder, lm_head, tokenizer):
        super().__init__(config)
        self.encoder = encoder
        self.audio_proj = AudioProjection()
        self.decoder = decoder
        self.lm_head = lm_head
        self.tokenizer = tokenizer

    def forward(self, audio_values, decoder_input_ids, labels=None):
        attention_mask = (audio_values != 0).long()

        encoder_out = self.encoder(
            audio_values,
            attention_mask=attention_mask,
        ).last_hidden_state

        projected = self.audio_proj(encoder_out)

        decoder_out = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=projected,
        ).last_hidden_state

        logits = self.lm_head(decoder_out)

        if labels is not None:
            # Standard cross-entropy at the decoder output
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            return loss, logits

        return logits

    def generate(self, audio_values, attention_mask=None, max_new_tokens=50):
        with torch.no_grad():
            encoder_out = self.encoder(
                audio_values, attention_mask=attention_mask
            ).last_hidden_state

            projected = self.audio_proj(encoder_out)

            out = torch.zeros(
                audio_values.size(0), 1, dtype=torch.long, device=audio_values.device
            )

            # Autoregressive inference
            for _ in range(max_new_tokens):
                hidden = self.decoder(
                    input_ids=out, encoder_hidden_states=projected
                ).last_hidden_state

                next_token = self.lm_head(hidden[:, -1:]).argmax(-1)

                out = torch.cat([out, next_token], dim=1)

                if (next_token == self.tokenizer.eos_token_id).all():
                    break

        return out[:, 1:]


def build_model(tokenizer):
    model = KoToNe(PretrainedConfig(), encoder, decoder, lm_head, tokenizer).to(device)

    # Freeze encoder entirely first
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Unfreeze top 6 transformer layers of the encoder
    # XLS-R (wav2vec2-large) has 24 transformer layers (indexed 0-23)
    # Top 6 = layers 18-23
    num_encoder_layers = len(model.encoder.encoder.layers)  # should be 24
    unfreeze_top_n = 6

    for i, layer in enumerate(model.encoder.encoder.layers):
        if i >= num_encoder_layers - unfreeze_top_n:
            for param in layer.parameters():
                param.requires_grad = True

    # Keep CNN feature extractor frozen regardless
    for param in model.encoder.feature_extractor.parameters():
        param.requires_grad = False
    for param in model.encoder.feature_projection.parameters():
        param.requires_grad = False

    # Freeze decoder except for cross-attention
    for name, param in model.decoder.named_parameters():
        if "EncDecAttention" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Calculate trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(
        f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
    )

    return model
