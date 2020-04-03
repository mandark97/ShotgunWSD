from abc import ABC, abstractmethod


class SynsetRelatedness(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def compute_similarity(self, word1, synset1, word2, synset2) -> float:
        pass
