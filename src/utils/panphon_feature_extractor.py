import panphon
import torch


class PanPhonFeatureExtractor:
    ft = panphon.FeatureTable()

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        # Get feature matrix over the entire tokenzier vocabulary
        self.vocab_matrix = self.feature_matrix_from_list(tokenizer.vocab)
        print(f"Tokenizer vocabulary: {tokenizer.vocab}")
        print(f"Matrix shape: {self.vocab_matrix.shape}")
        print(self.vocab_matrix)

        # TODO: Can map {-1, 0, 1} to {0, 0.5, 1}?

    def feature_vector_from_char(self, char):
        """
        Returns, e.g., [[1, 1, -1, ... 0, 0]] containing all 24 features
        If token is blank/has no articulatory features, returns []
        """
        char_features = self.ft.word_array([], char)
        return char_features.tolist()

    def feature_matrix_from_list(self, list):
        feature_matrix = torch.full((36, 24), 0, dtype=torch.float32)

        for index, token in enumerate(list):
            features = self.feature_vector_from_char(token)

            # Ignore unknown token and those without articulatory features
            if token == "<unk>" or len(features) == 0:
                print("No features")
            else:
                feature_matrix[index] = torch.tensor(features[0])

        return feature_matrix

    def get_vocab_matrix(self):
        """Returns the feature matrix over the entire tokenizer vocab"""
        return self.vocab_matrix
