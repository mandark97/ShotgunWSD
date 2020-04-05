from typing import List

from nltk.corpus.reader import Synset

from main import WordSynsets


def do_stuff(param):
    pass


class SynsetUtils(object):

    @staticmethod
    def compute_configuration_score(synsets, scores_matrix) -> float:
        for word_index1, synset_index1 in enumerate(synsets):
            for word_index2, synset_index2 in enumerate(synsets[word_index1 + 1:], start=word_index1 + 1):
                a = do_stuff(scores_matrix[(word_index1, synset_index1, word_index2, synset_index2)])

        return 0.0

    @classmethod
    def get_synsets(cls, synset_indexes: List[int], word_synsets: WordSynsets) -> List[Synset]:
        return [word_synsets[word_index][1][synset_index] for word_index, synset_index in enumerate(synset_indexes)]
