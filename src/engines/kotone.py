import torch
import torch.nn as nn
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    T5ForConditionalGeneration,
    Wav2Vec2Model,
)

from src.engines.fusion import BYT5_HIDDEN, CONV_DOWNSAMPLE_FACTOR, AudioProjection

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


def get_output_lengths(input_lengths, wav2vec_stride, conv_downsample_factor):
    # Account for wav2vec2's internal downsampling, then AudioProjection's conv
    encoder_lengths = input_lengths // wav2vec_stride
    return (encoder_lengths - 1) // conv_downsample_factor + 1


class KoToNe(PreTrainedModel):
    def __init__(self, config, encoder, decoder, lm_head, tokenizer):
        super().__init__(config)
        self.encoder = encoder
        self.audio_proj = AudioProjection()
        self.decoder = decoder
        self.lm_head = lm_head
        self.tokenizer = tokenizer
        self.ctc_head = nn.Linear(BYT5_HIDDEN, 384)
        self.ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

        # Query wav2vec2's total convolutional stride from the model itself
        self.wav2vec_stride = torch.prod(
            torch.tensor(self.encoder.config.conv_stride)
        ).item()

    def forward(self, audio_values, decoder_input_ids, labels=None, audio_lengths=None):
        attention_mask = (audio_values != 0).long()

        encoder_out = self.encoder(
            audio_values,
            attention_mask=attention_mask,
        ).last_hidden_state

        projected = self.audio_proj(encoder_out)

        # TODO: Go over this block again
        # CTC head
        ctc_logits = self.ctc_head(projected)  # (B, T', 1472) -> (B, T', 384)
        log_probs = ctc_logits.log_softmax(-1)  # (B, T', 384)
        log_probs = log_probs.transpose(0, 1)  # (T', B, 384)

        # Calculate the input size after downsampling
        if audio_lengths is not None:
            input_lengths = get_output_lengths(
                audio_lengths, self.wav2vec_stride, CONV_DOWNSAMPLE_FACTOR
            )
        else:
            # Fallback: assume no padding
            input_lengths = torch.full(
                (projected.size(0),),
                projected.size(1),
                dtype=torch.long,
                device=projected.device,
            )

        decoder_out = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=projected,
        ).last_hidden_state

        logits = self.lm_head(decoder_out)

        if labels is not None:
            # Standard cross-entropy at the decoder output
            ce_loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

            target_lengths = (labels != -100).sum(
                dim=1
            )  # (B,) — non-padded label lengths
            targets = labels[labels != -100]  # flattened, padding removed

            ctc_loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)

            # Combined loss
            loss = 0.8 * ce_loss + 0.2 * ctc_loss

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

    # Freeze encoder entirely
    for param in model.encoder.parameters():
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
