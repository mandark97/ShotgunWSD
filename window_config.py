from typing import List

from nltk.corpus.reader import Synset


class WindowConfiguration(object):
    def __init__(self, synset_indexes: List[int], window_words: List[str], window_words_pos: List[str],
                 configuration_synsets: List[Synset], score: float):
        self.synset_indexes = synset_indexes
        self.window_words = window_words
        self.window_words_pos = window_words_pos
        self.configuration_synsets = configuration_synsets
        self.score = score

    # TODO
    @staticmethod
    def has_collisions(window1, window2, offset, synset_collisions):
        pass

    # TODO
    @staticmethod
    def merge(window1, window2, offset):
        pass
