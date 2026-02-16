import panphon
import torch
import torch.nn as nn


class PanPhonFeatureExtractor:
    ft = panphon.FeatureTable()
    bce = nn.CrossEntropyLoss()

    def extract_features(self, sentence):
        words = sentence.split()
        word_features = {}

        for word in words:
            word_features[word] = self.ft.word_array([], word)

        for word, array in word_features.items():
            print(f"Word: {word}")
            print(f"Shape (Segments, Features): {array.shape}")
            # print(array)
            print("-" * 80)

        # TODO: Forced alignment

        return word_features

    def bce_loss(self, predicted, actual):
        predicted_features = self.extract_features(predicted)
        actual_features = self.extract_features(actual)
        loss = self.bce(predicted_features, actual_features)

        return loss


ppfe = PanPhonFeatureExtractor()
loss = ppfe.bce_loss(
    "ʔaŋ talumpati aj isaŋ uɾi ŋ kompetiʃon ŋ mɡa paɡbasa.",
    "ʔaŋ talumpataj isaŋ uɾi ŋ kompitʃon ŋ mɡa paɡbasa.",
)
