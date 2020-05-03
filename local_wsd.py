import logging
from typing import List, Tuple, Dict

from nltk.corpus.reader import Synset

from synset_relatedness import SynsetRelatedness
from synset_utils import SynsetUtils
from window_config import WindowConfiguration

ScoreMatrix = Dict[Tuple[int, int, int, int], float]


class LocalWSD(object):
    similarity_matrix: Dict[Tuple[int, int, int, int], float]
    word_synsets: List[Tuple[str, List[Synset]]]

    # Maybe combine window_words and window_words_pos to a tuple
    def __init__(self, word_offset: int, window_words: List[str], window_words_pos: List[str],
                 window_words_lemma: List[str], number_configs: int, synset_relatedness: SynsetRelatedness):
        self.word_offset = word_offset
        self.window_words = window_words
        self.window_words_pos = window_words_pos
        self.window_words_lemma = window_words_lemma
        self.number_configs = number_configs
        self.synset_relatedness = synset_relatedness

        self.windows_solutions: List[WindowConfiguration] = []

        super().__init__()

    def run(self):
        logging.debug(f"Starting for word_index: {self.word_offset}")
        self.word_synsets = self.build_window_synsets_array()
        self.similarity_matrix = self.compute_relatedness(self.word_synsets)

        self.generate_synset_combinations()

    def build_window_synsets_array(self) -> List[Tuple[str, List[Synset]]]:
        word_synsets = []
        # synsets_len = {} # TODO: Do we need this variable?
        for index, word in enumerate(self.window_words):
            synsets = SynsetUtils.get_wordnet_synsets(word, pos=self.window_words_pos[index],
                                                      lemma=self.window_words_lemma[index])
            # synsets_len[index] = len(synsets)
            if len(synsets) == 0:
                word_synsets.append((word, [None]))
            else:
                word_synsets.append((word, synsets))

        return word_synsets

    def compute_relatedness(self, word_synsets: List[Tuple[str, List[Synset]]]) -> ScoreMatrix:
        logging.debug("Start relatedness computing")
        similarity_matrix = {}

        for word1_index, (word1, synsets1) in enumerate(word_synsets):
            for synset1_index, synset1 in enumerate(synsets1):
                for word2_index, (word2, synsets2) in enumerate(word_synsets[word1_index + 1:], start=word1_index + 1):
                    for synset2_index, synset2 in enumerate(synsets2):
                        logging.debug(f"Compute relatedness for {word1}, {synset1}, {word2}, {synset2}")
                        sim = self.synset_relatedness.compute_similarity(word1, synset1, word2, synset2)
                        similarity_matrix[(word1_index, synset1_index, word2_index, synset2_index)] = sim
                        similarity_matrix[(word2_index, synset2_index, word1_index, synset1_index)] = sim

        logging.debug("Finished computing relatedness matrix")
        return similarity_matrix

    def generate_synset_combinations(self):
        """
        Generate best synset combinations with the highest scores
        """
        self.combinations_recursion(0, [0] * len(self.window_words))
        if len(self.windows_solutions) == 0:
            self.windows_solutions = None
        else:
            for window_configuration in self.windows_solutions:
                window_configuration.set_global_ids(self.word_offset)

        logging.debug("Finished generation synset_combinations")

    def combinations_recursion(self, word_index: int, synset_indexes: List[int]):
        """
        Recursion to go through the all combinations
        :param word_index: Index of the word we are at
        :param synset_indexes: Array of the index of the sense we are choosing
        """
        for synset_index, synset in enumerate(self.word_synsets[word_index][1]):
            synset_indexes[word_index] = synset_index

            if word_index < len(self.word_synsets) - 1:
                self.combinations_recursion(word_index + 1, synset_indexes)
            else:
                score = SynsetUtils.compute_configuration_score(synset_indexes, self.similarity_matrix)
                configuration_synsets: List[Synset] = SynsetUtils.get_synsets(synset_indexes, self.word_synsets)

                if len(self.windows_solutions) >= self.number_configs:
                    if score >= self.windows_solutions[-1].get_score():
                        self.windows_solutions.append(WindowConfiguration(synset_indexes, self.window_words,
                                                                          self.window_words_pos, configuration_synsets,
                                                                          score))

                        sorted(self.windows_solutions, key=lambda window: window.get_score(), reverse=True)
                        self.windows_solutions.pop()
                else:
                    self.windows_solutions.append(WindowConfiguration(synset_indexes, self.window_words,
                                                                      self.window_words_pos, configuration_synsets,
                                                                      score))
                    if len(self.windows_solutions) == self.number_configs:
                        sorted(self.windows_solutions, key=lambda window: window.get_score(), reverse=True)
