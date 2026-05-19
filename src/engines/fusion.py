import torch.nn as nn

WAV2VEC_HIDDEN = 1024
BYT5_HIDDEN = 1472
CONV_DOWNSAMPLE_FACTOR = 2


# Projection bridge between wav2vec2 and ByT5
class AudioProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                WAV2VEC_HIDDEN,
                WAV2VEC_HIDDEN,
                kernel_size=3,
                stride=CONV_DOWNSAMPLE_FACTOR,
                padding=1,
            ),
            nn.GELU(),
        )
        self.proj = nn.Linear(WAV2VEC_HIDDEN, BYT5_HIDDEN)

    def forward(self, x):
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.proj(x)
