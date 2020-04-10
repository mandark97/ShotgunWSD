from typing import List, Tuple, Optional

from nltk.corpus.reader import Synset


class WindowConfiguration(object):
    global_synsets: List[Tuple[int, int]]

    def __init__(self, synset_indexes: List[int], window_words: List[str], window_words_pos: List[str],
                 configuration_synsets: List[Synset], score: float, global_synsets: List[Tuple[int, int]] = None):
        """
        Configuration of the window
        :param synset_indexes: Indexes of the senses chosen for this configuration
        :param window_words: Words in the windows
        :param window_words_pos: POS of words in the window
        :param configuration_synsets: Synsets of the window
        :param score: Score of the window
        :param global_synsets: Tuples of word_index and sense_index
        """
        self.synset_indexes = synset_indexes
        self.window_words = window_words
        self.window_words_pos = window_words_pos
        self.configuration_synsets = configuration_synsets
        self.score = score

        if global_synsets != None:
            self.global_synsets = global_synsets
            self.first_global_sense = self.global_synsets[0][0]
            self.last_global_sense = self.global_synsets[-1][0]

    def set_global_ids(self, window_offset: int):
        self.global_synsets = []
        for word_index, synset_index in enumerate(self.synset_indexes):
            self.global_synsets.append((window_offset + word_index, synset_index))

        self.first_global_sense = self.global_synsets[0][0]
        self.last_global_sense = self.global_synsets[-1][0]

    def contains_global_sense(self, word_id: int):
        return word_id >= self.first_global_sense and word_id <= self.last_global_sense

    def __len__(self):
        return len(self.synset_indexes)

    # TODO
    @staticmethod
    def has_collisions(window1, window2, offset, synset_collisions):
        pass

    # TODO
    @staticmethod
    def merge(window1, window2, offset):
        pass


def compare_by_length_and_value(window_config1: WindowConfiguration, window_config2: WindowConfiguration):
    if len(window_config1) == len(window_config2):
        if window_config1.score == window_config2.score:
            return 0
        else:
            return 1 if window_config1.score > window_config2.score else -1
    else:
        return 1 if len(window_config1) > len(window_config2) else -1
