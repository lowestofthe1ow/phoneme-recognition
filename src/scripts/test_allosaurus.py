from allosaurus.app import read_recognizer

model = read_recognizer()

# Example file
# TODO: Do more testing
print(model.recognize("data/magichub/asr-sfdusc/WAV/G0007/G0007_1_S0076.wav"))
