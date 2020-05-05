from typing import List, Tuple

from nltk.corpus.reader import Synset

from synset_utils import SynsetUtils


class WindowConfiguration(object):
    global_synsets: List[Tuple[int, int]]

    def __init__(self, synset_indexes: List[int], window_words: List[str], window_words_pos: List[str],
                 configuration_synsets: List[Synset], score: float = -1, global_synsets: List[Tuple[int, int]] = None):
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

    def get_score(self):
        if self.score == -1:
            self.score = SynsetUtils.compute_configuration_scores(self.configuration_synsets, self.window_words,
                                                                  self.global_synsets)
        return self.score

    def __len__(self):
        return len(self.synset_indexes)

    def __repr__(self):
        return repr((len(self.synset_indexes), self.score, self.global_synsets))

    @staticmethod
    def has_collisions(window1: "WindowConfiguration", window2: "WindowConfiguration", offset: int,
                       synset_collisions: int):
        for i in range(synset_collisions):
            if window1.synset_indexes[i - offset] != window2.synset_indexes[i]:
                return False
        return True

    @staticmethod
    def merge(window1: "WindowConfiguration", window2: "WindowConfiguration", offset: int):
        if window2.last_global_sense < window1.last_global_sense:
            return

        start_at = len(window1.synset_indexes) - offset
        synsets_indexes = window1.synset_indexes + window2.synset_indexes[start_at:]
        global_senses = window1.global_synsets + window2.global_synsets[start_at:]
        window_words = window1.window_words + window2.window_words[start_at:]
        window_words_pos = window1.window_words_pos + window2.window_words_pos[start_at:]
        configuration_synsets = window1.configuration_synsets + window2.configuration_synsets[start_at:]

        return WindowConfiguration(synsets_indexes, window_words, window_words_pos, configuration_synsets, -1,
                                   global_senses)


def compare_by_length_and_value(window_config1: WindowConfiguration, window_config2: WindowConfiguration):
    if len(window_config1) == len(window_config2):
        if window_config1.get_score() == window_config2.get_score():
            return 0
        else:
            return 1 if window_config2.get_score() > window_config1.get_score() else -1
    else:
        return 1 if len(window_config2) > len(window_config1) else -1
