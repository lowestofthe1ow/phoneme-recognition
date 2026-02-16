from src.utils.panphon_feature_extractor import PanPhonFeatureExtractor
import torch
import pysdtw
from nemo.utils import logging


class AugmentedCTCLoss(torch.nn.Module):
    def __init__(self, ctc_loss, tokenizer):
        super().__init__()
        self.ctc_loss = ctc_loss
        self.tokenizer = tokenizer
        self.ppfe = PanPhonFeatureExtractor(tokenizer)

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        """Augments the CTC loss with the SDTW loss"""
        ctc_loss = self.ctc_loss(
            log_probs=log_probs,
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
        )

        probs = torch.exp(log_probs).to("cuda")

        logging.info(f"{probs}")
        logging.info(f"Logits tensor shape: {probs.shape}")
        logging.info(f"Target tensor shape: {targets.shape}")

        # Gets feature representations for both predicted and target...
        feature_matrix = self.ppfe.get_vocab_matrix().to("cuda")
        predicted_features = torch.matmul(probs, feature_matrix)
        target_features = torch.nn.functional.embedding(targets, feature_matrix)

        # Tensors will be B * S * 24 for batch size B and sequence length S.
        # S can differ between predicted and target, but SDTW should handle it.
        # TODO: SDTW will suck at dealing with blank tokens
        logging.info(predicted_features.shape)
        logging.info(target_features.shape)

        # TODO: Is SDTW actually contributing a noticeable gradient?
        sdtw = pysdtw.SoftDTW(gamma=0.1, use_cuda=True)
        res = sdtw(predicted_features, target_features)
        sdtw_loss = res.mean()

        print(sdtw_loss)
        print(ctc_loss)

        return ctc_loss + sdtw_loss
