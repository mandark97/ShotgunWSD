from typing import List


class Document(object):
    def __init__(self, id: int, words: List[str], words_pos: List[str], words_lemma: List[str]):
        self.id = id
        self.words = words
        self.words_pos = words_pos
        self.words_lemma = words_lemma
        super().__init__()

    def __len__(self):
        return len(self.words)
