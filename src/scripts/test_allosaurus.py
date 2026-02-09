from allosaurus.app import read_recognizer

# load your model
model = read_recognizer()

# run inference -> æ l u s ɔ ɹ s
print(model.recognize('data/magichub/asr-sfdusc/WAV/G0007/G0007_1_S0076.wav'))
