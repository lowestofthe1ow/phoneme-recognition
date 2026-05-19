import torch
import torch.nn as nn
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    T5ForConditionalGeneration,
    Wav2Vec2Model,
)

from src.engines.fusion import AudioProjection

G2P_CHECKPOINT = "models/g2p/checkpoint-9670"


device = "cuda" if torch.cuda.is_available() else "cpu"

# Load wav2vec encoder
encoder = Wav2Vec2Model.from_pretrained(
    "Khalsuu/filipino-wav2vec2-l-xls-r-300m-official"
).to(device)

print(f"Encoder output dim: {encoder.config.hidden_size}")


# Load pretrained G2P
g2p = T5ForConditionalGeneration.from_pretrained(
    G2P_CHECKPOINT,
)

decoder = g2p.decoder  # Decoder
lm_head = g2p.lm_head  # Linear head on top of the decoder

# Freeze encoder
for param in decoder.parameters():
    param.requires_grad = False

# Get rid of the rest
del g2p
torch.cuda.empty_cache()


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

        with torch.no_grad():
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
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            return loss, logits

        return logits

    def generate(self, audio_values, attention_mask=None, max_new_tokens=50):
        with torch.no_grad():
            projected = self.audio_proj(
                self.encoder(
                    audio_values, attention_mask=attention_mask
                ).last_hidden_state
            )
            out = torch.zeros(
                audio_values.size(0), 1, dtype=torch.long, device=audio_values.device
            )
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

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
    )

    def print_grad(grad):
        print(f"Proj weight grad norm: {grad.norm():.4f}")

    # model.audio_proj.proj.weight.register_hook(print_grad)

    return model
