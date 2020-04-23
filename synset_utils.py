from typing import List, Dict, Tuple

from nltk.corpus.reader import Synset

from word_synsets import WordSynsets
from operations import ConfigurationOperation
from synset_relatedness import SynsetRelatedness


class SynsetUtils(object):
    # TODO global parameters to be set in main
    configuration_operation: ConfigurationOperation = None
    synset_relatedness: SynsetRelatedness = None
    cache_synset_relatedness: Dict[str, float] = {}

    @staticmethod
    def compute_configuration_score(synsets: List[int], scores_matrix: Dict[Tuple[int, int, int, int], float]) -> float:
        sense_score = SynsetUtils.configuration_operation.initial_score
        for word_index1, synset_index1 in enumerate(synsets):
            for word_index2, synset_index2 in enumerate(synsets[word_index1 + 1:], start=word_index1 + 1):
                sense_score += max(
                    SynsetUtils.configuration_operation.apply_operation(sense_score, scores_matrix[(word_index1, synset_index1, word_index2, synset_index2)]),
                    SynsetUtils.configuration_operation.apply_operation(sense_score, scores_matrix[(word_index2, synset_index2, word_index1, synset_index1)])
                )

        return sense_score

    @staticmethod
    def compute_configuration_scores(synsets: List[int], words: List[str], global_synsets: List[Tuple[int, int]]) -> float:
        sense_score = SynsetUtils.configuration_operation.initial_score
        for i in range(len(synsets) - 1):
            target_synset = synsets[i]
            target_word = words[i]
            target_global_sense = global_synsets[i]
            for j in range(i+1, len(synsets)):
                key1 = target_global_sense + "||" + global_synsets[j]
                key2 = global_synsets[j] + "||" + target_global_sense
                if key1 in SynsetUtils.cache_synset_relatedness:
                    score = SynsetUtils.cache_synset_relatedness[key1]
                elif key2 in SynsetUtils.cache_synset_relatedness:
                    score = SynsetUtils.cache_synset_relatedness[key2]
                else:
                    score = SynsetUtils.synset_relatedness.compute_similarity(
                        target_word, target_synset, words[j], synsets[j])
                    SynsetUtils.cache_synset_relatedness[key1] = score

                sense_score = SynsetUtils.configuration_operation.apply_operation(sense_score, score)
        return sense_score

    @classmethod
    def get_synsets(cls, synset_indexes: List[int], word_synsets: WordSynsets) -> List[Synset]:
        return [word_synsets[word_index][1][synset_index] for word_index, synset_index in enumerate(synset_indexes)]
