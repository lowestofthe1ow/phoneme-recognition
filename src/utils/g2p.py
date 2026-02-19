import epitran
from lingua import Language, LanguageDetectorBuilder


class G2P:
    def __init__(self):
        # Set up Epitran
        self.epi_en = epitran.Epitran("eng-Latn")
        self.epi_tl = epitran.Epitran("tgl-Latn")

        # Set up Lingua
        languages = [Language.ENGLISH, Language.TAGALOG]
        self.detector = LanguageDetectorBuilder.from_languages(*languages).build()

    def transliterate(self, sentence):
        out = []

        for word in sentence.split():
            lang = self.detector.detect_language_of(word)

            if lang.name == "ENGLISH":
                result = self.epi_en.transliterate(word)
            else:
                result = self.epi_tl.transliterate(word)

            out.append(result)

        return " ".join(out)
