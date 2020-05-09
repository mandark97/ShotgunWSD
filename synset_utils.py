from typing import List, Dict, Tuple

from nltk.corpus import wordnet as wn
from nltk.corpus.reader import Synset

from operations import ConfigurationOperation
from relatedness.synset_relatedness import SynsetRelatedness

POS_MAPPING = {
    "NOUN": wn.NOUN,
    "VERB": wn.VERB,
    "ADV": wn.ADV,
    "ADJ": wn.ADJ,
}


def get_pos(pos):
    return POS_MAPPING.get(pos, '')


class SynsetUtils(object):
    # TODO global parameters to be set in main
    configuration_operation: ConfigurationOperation = None
    synset_relatedness: SynsetRelatedness = None
    cache_synset_relatedness: Dict[Tuple[Tuple[int, int], Tuple[int, int]], float] = {}

    @staticmethod
    def compute_configuration_score(synsets: List[int], scores_matrix: Dict[Tuple[int, int, int, int], float]) -> float:
        sense_score = SynsetUtils.configuration_operation.initial_score
        for word_index1, synset_index1 in enumerate(synsets):
            for word_index2, synset_index2 in enumerate(synsets[word_index1 + 1:], start=word_index1 + 1):
                sense_score = SynsetUtils.configuration_operation.apply_operation(sense_score, scores_matrix[
                    (word_index1, synset_index1, word_index2, synset_index2)])
                sense_score = SynsetUtils.configuration_operation.apply_operation(sense_score, scores_matrix[
                    (word_index2, synset_index2, word_index1, synset_index1)])
        return sense_score

    @staticmethod
    def compute_configuration_scores(synsets: List[Synset], words: List[str],
                                     global_synsets: List[Tuple[int, int]]) -> float:
        sense_score = SynsetUtils.configuration_operation.initial_score
        for i in range(len(synsets) - 1):
            target_synset = synsets[i]
            target_word = words[i]
            target_global_sense = global_synsets[i]
            for j in range(i + 1, len(synsets)):
                key1 = (target_global_sense, global_synsets[j])
                key2 = (global_synsets[j], target_global_sense)
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

    @staticmethod
    def get_synsets(synset_indexes: List[int], word_synsets: List[Tuple[str, List[Synset]]]) -> List[Synset]:
        return [word_synsets[word_index][1][synset_index] for word_index, synset_index in enumerate(synset_indexes)]

    @staticmethod
    def sense_key(synset: Synset, lemma: str):
        for l in synset.lemmas():
            if l.name().lower() == lemma.lower():
                return l.key()
        print(f"No sense key found for {lemma} and {synset}")
        return synset.lemmas()[0].key()

    @staticmethod
    def get_wordnet_synsets(word, pos, lemma=None):
        word_pos = get_pos(pos)
        # synsets = [synset for synset in wn.synsets(word, pos=word_pos)]

        # if len(synsets) == 0:
        synsets = [l.synset() for l in wn.lemmas(lemma) if l.synset().pos() == word_pos]

        return synsets
