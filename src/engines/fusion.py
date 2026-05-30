import torch.nn as nn

WAV2VEC_HIDDEN = 1024
BYT5_HIDDEN = 1472
CONV_KERNEL_SIZE = 3
CONV_DOWNSAMPLE_FACTOR = 2


class AudioProjection(nn.Module):
    """Bridges the wav2vec2-based encoder with the ByT5-based decoder.

    Input sequence is downsampled with a 1D convolution, layer-normalized, then
    performs a linear projection from wav2vec2 hidden size (1024) to ByT5
    hidden size (1472).
    """

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(
                WAV2VEC_HIDDEN,
                WAV2VEC_HIDDEN,
                kernel_size=CONV_KERNEL_SIZE,
                stride=CONV_DOWNSAMPLE_FACTOR,
                padding=1,
            ),
            nn.GELU(),
        )

        self.norm = nn.LayerNorm(WAV2VEC_HIDDEN)
        self.proj = nn.Linear(WAV2VEC_HIDDEN, BYT5_HIDDEN)

        # Near-identity initialization for stable training
        nn.init.normal_(self.proj.weight, mean=0.0, std=1e-4)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        # HuggingFace Transformers uses (B, S, D), so transpose to (S, B, D)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)

        # Pre-norm then project
        x = self.norm(x)
        return self.proj(x)
