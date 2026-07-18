import torch.nn as nn
import torch.nn.functional as F

from src.engines.raripa import RARIPA


class RARIPACTC(RARIPA):
    def __init__(self):
        super().__init__()

        # CTC head projecting to ByT5 vocabulary
        # zero_infinity=True works as a safety net for when characters > frames
        self.ctc_head = nn.Linear(self.config.d_model, self.config.vocab_size + 1)
        self.ctc_loss = nn.CTCLoss(blank=self.config.vocab_size, zero_infinity=True)
        self.ctc_weight = 0.3

    def forward(
        self,
        audio_values,
        labels=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        **kwargs,
    ):
        # Grab the projected inputs to ByT5 as well as the downsampled mask
        # TODO: Is this even necessary? Not even sure now what "skipping ByT5"
        # should look like, lol.
        projected, mask, projected_lengths, _, feat_lengths = self._get_projected(
            audio_values, attention_mask
        )

        # Grab CTC logits and loss if training with it
        if labels is not None:
            ctc_logits = self.ctc_head(projected)

            # PyTorch CTCLoss expects (Seq, Batch, Vocab) with log probabilities
            log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)

            target_lengths = (labels != -100).sum(dim=-1)
            ctc_targets = labels.clone()
            ctc_targets[ctc_targets == -100] = self.config.pad_token_id

            ctc_loss = self.ctc_loss(
                log_probs,
                ctc_targets,
                input_lengths=projected_lengths,
                target_lengths=target_lengths,
            )

        # Bypass ByT5 entirely if training only on CTC loss
        return {"loss": ctc_loss, "logits": ctc_logits, "hidden_states": projected}
