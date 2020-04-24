from typing import List, Optional


class Document(object):
    def __init__(self, id: int, words: List[str], words_pos: List[str], words_lemma: List[str],
                 words_id: List[Optional[str]]):
        self.id = id
        self.words = words
        self.words_pos = words_pos
        self.words_lemma = words_lemma
        self.words_id = words_id
        super().__init__()

    def __len__(self):
        return len(self.words)
