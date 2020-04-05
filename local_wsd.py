from typing import List, Tuple, Dict, Optional

from nltk.corpus.reader import Synset

from pos_utils import get_pos
from synset_relatedness import SynsetRelatedness
from synset_utils import SynsetUtils
from window_config import WindowConfiguration
from nltk.corpus import wordnet as wn

WordSynsets = List[Tuple[str, List[Synset]]]
ScoreMatrix = Dict[Tuple[int, int, int, int], float]


class LocalWSD(object):
    windows_solutions: Optional[List[WindowConfiguration]]

    # Maybe combine window_words and window_words_pos to a tuple
    def __init__(self, word_offset: int, window_words: List[str], window_words_pos: List[str], number_configs: int,
                 synset_relatedness: SynsetRelatedness):
        self.word_offset = word_offset
        self.window_words = window_words
        self.window_words_pos = window_words_pos
        self.number_configs = number_configs
        self.synset_relatedness = synset_relatedness

        self.windows_solutions = []

        super().__init__()

    def run(self):
        self.word_synsets = self.build_window_synsets_array()
        self.similarity_matrix = self.compute_relatedness(self.word_synsets)

        self.generate_synset_combinations()

    def build_window_synsets_array(self) -> WordSynsets:
        word_synsets = []
        synsets_len = {}
        for index, word in enumerate(self.window_words):
            synsets = wn.synsets(word, pos=get_pos(self.window_words_pos[index]))
            synsets_len[index] = len(synsets)
            if len(synsets) == 0:
                word_synsets.append((word, [None]))
            else:
                word_synsets.append((word, synsets))
                # for synset in synsets:
                #     word_synsets.append((word, synset))

        return word_synsets

    # def compute_relatedness(self, word_synsets: List[Tuple[str, Synset]]) -> List[List[float]]:
    #     word_synset_tuples = [[(word, synset) for synset in synsets] for (word, synsets) in word_synsets]
    #     similarity_matrix = [[0.0 for _ in range(len(word_synset_tuples))] for _ in range(len(word_synset_tuples))]
    #
    #     for i, (word1, synset1) in enumerate(word_synset_tuples):
    #         for j, (word2, synset2) in enumerate(word_synset_tuples[i:], start=i):
    #             sim = self.synset_relatedness.compute_similarity(word1, synset1, word2, synset2)
    #             similarity_matrix[i][j] = sim
    #             similarity_matrix[j][i] = sim
    #
    #     return similarity_matrix

    def compute_relatedness(self, word_synsets: WordSynsets) -> ScoreMatrix:
        similarity_matrix = {}

        for word1_index, (word1, synsets1) in enumerate(word_synsets):
            for synset1_index, synset1 in enumerate(synsets1):
                for word2_index, (word2, synsets2) in enumerate(word_synsets[word1_index:], start=word1_index):
                    for synset2_index, synset2 in enumerate(synsets2):
                        sim = self.synset_relatedness.compute_similarity(word1, synset1, word2, synset2)
                        similarity_matrix[(word1_index, synset1_index, word2_index, synset2_index)] = sim
                        similarity_matrix[(word2_index, synset2_index, word1_index, synset1_index)] = sim

        return similarity_matrix

    # get the best configurations
    def generate_synset_combinations(self):
        self.combinations_recursion(0, [0] * len(self.window_words))
        if len(self.windows_solutions) == 0:
            self.windows_solutions = None
        # TODO else set global ids

    def combinations_recursion(self, word_index: int, synset_indexes: List[int]):
        for synset_index, synset in enumerate(self.word_synsets[word_index][1]):
            synset_indexes[word_index] = synset_index

            if word_index < len(self.word_synsets) - 1:
                self.combinations_recursion(word_index + 1, synset_indexes)
            else:
                score = SynsetUtils.compute_configuration_score(synset_indexes, self.similarity_matrix)
                configuration_synsets: List[Synset] = SynsetUtils.get_synsets(synset_indexes, self.word_synsets)

                if len(self.windows_solutions) >= self.number_configs:
                    if score >= self.windows_solutions[-1].score:
                        self.windows_solutions.append(WindowConfiguration(synset_indexes, self.window_words,
                                                                          self.window_words_pos, configuration_synsets,
                                                                          score))

                        sorted(self.windows_solutions, key=lambda window: window.score, reverse=True)
                        self.windows_solutions.pop()
                else:
                    self.windows_solutions.append(WindowConfiguration(synset_indexes, self.window_words,
                                                                      self.window_words_pos, configuration_synsets,
                                                                      score))
                    if len(self.windows_solutions) == self.number_configs:
                        sorted(self.windows_solutions, key=lambda window: window.score, reverse=True)
