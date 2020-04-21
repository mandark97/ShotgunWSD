from abc import ABC, abstractmethod
from nltk.corpus.reader.wordnet import Synset
from nltk.tokenize import PunktSentenceTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords


class SynsetRelatedness(ABC):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_gloss(synset: Synset):
        return " ".join(synset.examples()) + " " + synset.definition()

    @staticmethod
    def preprocess_text(text: str) -> list[str]:
        # tokenize
        tokens = PunktSentenceTokenizer().tokenize(text)
        # remove stopwords
        english_sw = stopwords.words("english")
        tokens = [w for w in tokens if w not in english_sw]
        # apply stemming
        stemmer = SnowballStemmer("english")
        tokens = [stemmer.stem(w) for w in tokens]
        return tokens

    @abstractmethod
    def compute_similarity(self, word1, synset1, word2, synset2) -> float:
        pass
