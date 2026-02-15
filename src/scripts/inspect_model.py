import argparse

import nemo.collections.asr as nemo_asr
from torchinfo import summary

parser = argparse.ArgumentParser(description="Inspects a .nemo model at a given path.")
parser.add_argument("path", help="The input .nemo model file path")
args = parser.parse_args()

model = nemo_asr.models.ASRModel.restore_from(args.path)

summary(model)
