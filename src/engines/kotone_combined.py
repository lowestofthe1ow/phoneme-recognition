import torch.nn as nn
import torch.nn.functional as F

from src.engines.kotone import KoToNe


class KoToNeCombined(KoToNe):
    def __init__(self):
        super().__init__()

        # CTC head projecting to ByT5 vocabulary
        # zero_infinity=True works as a safety net for when characters > frames
        self.ctc_head = nn.Linear(1024, self.config.vocab_size + 1)
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
        projected, mask, projected_lengths, encoder_out = self._get_projected(
            audio_values, attention_mask
        )

        # Delegate to the forward() call in the ByT5 model bypassing its encoder
        # entirely, using our projected inputs instead
        outputs = self.byt5(
            encoder_outputs=(projected,),
            attention_mask=mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            **kwargs,
        )

        # Grab CTC logits and loss if training with it
        if labels is not None:
            ctc_logits = self.ctc_head(encoder_out)

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

            # Combine losses
            ce_loss = outputs.loss

            print(f"CE: {ce_loss}; CTC: {ctc_loss}")

            outputs.loss = (self.ctc_weight * ctc_loss) + (
                (1 - self.ctc_weight) * ce_loss
            )

        return outputs
