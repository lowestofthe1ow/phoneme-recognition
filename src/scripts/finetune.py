import nemo.collections.asr as nemo_asr

from torchinfo import summary

# model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")
model = nemo_asr.models.ASRModel.restore_from("models/nvidia/parakeet-tdt-0.6b-v3.nemo")

summary(model)

files = ["data/nexdata/filipino_822/G00001/G00001S0618.wav"]

transcriptions = model.transcribe(files)

print(transcriptions)

"""
print("=" * 85)

summary(model)

print("=" * 85)

print(model.encoder)

print("=" * 85)

print(model.decoder)


# files = ["data/samples/OSR_us_000_0010_8k.wav"]

# transcriptions = model.transcribe(files)

# print(transcriptions)
"""
