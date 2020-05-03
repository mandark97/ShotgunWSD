import logging
from collections import defaultdict
from functools import reduce, cmp_to_key
from math import log
from typing import List, Dict, Tuple, Optional, DefaultDict

from nltk.corpus.reader import Synset

from local_wsd import LocalWSD
from parsed_document import Document
from synset_relatedness import SynsetRelatedness
from synset_utils import SynsetUtils
from window_config import WindowConfiguration, compare_by_length_and_value


class ShotgunWSD(object):
    def __init__(self, document: Document, window_size: int, number_configs: int, synset_relatedness: SynsetRelatedness,
                 min_synset_collision: int, max_synset_collision: int, number_of_votes: int):
        self.document = document
        self.window_size = window_size
        self.number_configs = number_configs
        self.synset_relatedness = synset_relatedness
        self.min_synset_collision = min_synset_collision
        self.max_synset_collision = max_synset_collision
        self.number_of_votes = number_of_votes

        super().__init__()

    def run(self) -> List[Optional[Synset]]:
        logging.debug("Run algorithm")
        document_window_solutions = self.compute_windows()
        logging.debug(f"Found {len(document_window_solutions)} windows")
        document_window_solutions = self.merge_window_solutions(document_window_solutions)

        sense_votes = self.vote_senses(document_window_solutions)
        senses = self.select_senses(document_window_solutions, sense_votes)
        finalSenses = self.detect_most_used_senses(senses)
        convertedSynsets = self.convertFinalSynsets(finalSenses)

        return convertedSynsets

    def compute_windows(self) -> Dict[int, List[WindowConfiguration]]:
        logging.debug("Compute windows")
        document_window_solutions = {}

        for word_index in range(0, len(self.document.words) - self.window_size):
            window_words = self.document.words[word_index:word_index + self.window_size]
            window_words_pos = self.document.words_pos[word_index:word_index + self.window_size]
            window_words_lemma = self.document.words_lemma[word_index:word_index + self.window_size]

            # TODO some max synset combination black magic
            local_wsd = LocalWSD(word_offset=word_index, window_words=window_words, window_words_pos=window_words_pos,
                                 window_words_lemma=window_words_lemma, number_configs=self.number_configs,
                                 synset_relatedness=self.synset_relatedness)
            local_wsd.run()

            document_window_solutions[word_index] = local_wsd.windows_solutions

        return document_window_solutions

    def merge_window_solutions(self, document_window_solutions) -> Dict[int, List[WindowConfiguration]]:
        logging.debug(f"Start merging {len(document_window_solutions)} window solutions")
        merged_windows = None

        for synset_collisions in reversed(range(self.min_synset_collision, self.max_synset_collision + 1)):
            merged_windows = self.merge_windows(document_window_solutions, synset_collisions)

        logging.debug(f"Obtained {len(merged_windows)}")
        return merged_windows

    # works?
    def merge_windows(self, document_window_solutions: Dict[int, List[WindowConfiguration]],
                      synset_collisions: int) -> Dict[int, List[WindowConfiguration]]:
        for l in range(len(self.document.words)):
            if l not in document_window_solutions:
                continue

            config_list1 = document_window_solutions[l]
            for window1 in config_list1:
                for j in range(len(window1.synset_indexes) - synset_collisions):
                    if j + l + 1 not in document_window_solutions:
                        continue

                    config_list2 = document_window_solutions[j + l + 1]
                    for window2 in config_list2:
                        # collided = False
                        if WindowConfiguration.has_collisions(window1, window2, j + 1, synset_collisions):
                            logging.debug(f"merging {window1} and {window2}")
                            merged_window = WindowConfiguration.merge(window1, window2, j + 1)
                            if merged_window != None:
                                # collided = True
                                config_list1.append(merged_window)
                            config_list2.remove(window2)

        return document_window_solutions

    def vote_senses(self, document_window_solutions: Dict[int, List[WindowConfiguration]]) -> List[
        Optional[Tuple[int, int]]]:
        all_windows: List[WindowConfiguration] = reduce(lambda x, y: x + y, document_window_solutions.values())
        sorted(all_windows, key=len, reverse=True)
        word_sense_weights = self.compute_word_sense_weights(all_windows)

        max_weights: List[float] = [0.0] * len(self.document)
        results: List[Optional[Tuple[int, int]]] = [None] * len(self.document)
        for word_index in range(len(self.document)):
            if word_index in word_sense_weights:
                tmp: Dict[Tuple[int, int], float] = word_sense_weights[word_index]

                for synset_index, synset_weight in tmp.items():
                    if synset_weight > max_weights[word_index]:
                        results[word_index] = synset_index
                        max_weights[word_index] = synset_weight
                    elif synset_weight == max_weights[word_index]:
                        results[word_index] = None  # ?

        return results

    def compute_word_sense_weights(self, wsd_windows: List[WindowConfiguration]) -> Dict[
        int, Dict[Tuple[int, int], float]]:
        """

        :param wsd_windows: Configuration windows sorted by length
        :return: For each word, the weight for each sense
        """
        word_sense_weights: Dict[int, Dict[Tuple[int, int], float]] = {}
        for word_index in range(len(self.document)):
            no_of_windows: int = 0

            # Get all windows that contain the current word_index
            temp_indexed_list = list(filter(lambda window: window.contains_global_sense(word_index), wsd_windows))
            temp_indexed_list = self.extract_wsd_windows(temp_indexed_list)

            sorted(temp_indexed_list, key=cmp_to_key(compare_by_length_and_value))

            for wsd in temp_indexed_list:
                if no_of_windows == self.number_of_votes:
                    break

                if wsd.contains_global_sense(word_index) != True:  # ? useless?
                    continue

                weight = log(len(wsd))
                no_of_windows += 1

                global_synset: Tuple[int, int] = wsd.global_synsets[word_index - wsd.first_global_sense]
                global_synset_word_id: int = global_synset[0]
                if global_synset_word_id in word_sense_weights:
                    tmp = word_sense_weights[global_synset_word_id]
                    if global_synset in tmp:
                        tmp[global_synset] = tmp[global_synset] + weight
                    else:
                        tmp[global_synset] = weight
                else:
                    tmp = {global_synset: weight}

                word_sense_weights[global_synset_word_id] = tmp

        return word_sense_weights

    def extract_wsd_windows(self, wsd_windows: List[WindowConfiguration]) -> List[WindowConfiguration]:
        """
        Get only the longest windows
        """
        tmp_size = 0
        returned_senses: List[WindowConfiguration] = []

        for wsd_window in wsd_windows:
            if tmp_size != len(wsd_window):
                if len(returned_senses) >= self.number_of_votes:
                    break

                tmp_size = len(wsd_windows)
            returned_senses.append(wsd_window)

        return returned_senses

    def select_senses(self, document_window_solutions: Dict[int, List[WindowConfiguration]],
                      sense_votes: List[Optional[Tuple[int, int]]]) -> List[Optional[Tuple[int, int]]]:
        final_synsets = [None] * len(self.document)
        synset_window_size = [0] * len(self.document)
        synset_window_score = [0.0] * len(self.document)

        window_solutions: List[WindowConfiguration]
        for word_index, window_solutions in document_window_solutions.items():
            if window_solutions is None or len(window_solutions) == 0:
                continue
            sorted(window_solutions, key=len, reverse=True)

            max_length = len(window_solutions[0])

            for wsd in window_solutions:
                if len(wsd) < max_length:
                    break

                for window_index, global_synset in enumerate(wsd.synset_indexes):
                    if sense_votes is None or sense_votes[word_index + window_index] is None:
                        if final_synsets[word_index + window_index] is None or \
                                len(wsd) >= synset_window_size[word_index + window_index] and \
                                wsd.get_score() > synset_window_score[word_index + window_index]:
                            final_synsets[word_index + window_index] = global_synset
                            synset_window_score[word_index + window_index] = len(wsd)
                            synset_window_score[word_index + window_index] = wsd.get_score()

                    else:
                        final_synsets[word_index + window_index] = sense_votes[word_index + window_index]

        return final_synsets

    def detect_most_used_senses(self, senses: List[Optional[Tuple[int, int]]]) -> List[Tuple[int, int]]:
        word_sense_count: DefaultDict[str, DefaultDict[int, int]] = defaultdict(lambda: defaultdict(int))

        for i in range(len(self.document)):
            if senses[i] is not None:
                synset_index = senses[i][1]
                word = f"{self.document.words[i]}||{self.document.words_pos[i]}"
                word_sense_count[word][synset_index] += 1

        # Remove words that appear with only one sense
        for i in range(len(self.document)):
            key = f"{self.document.words[i]}||{self.document.words_pos[i]}"
            if key in word_sense_count and len(word_sense_count[key]) == 1:
                word_sense_count.pop(key)

        final_word_sense_count: Dict[str, Tuple[int, int]] = {}
        for word_key, sense_count in word_sense_count.items():
            max_count = -1
            sense_idx = ""
            remove = False
            for sense_index, count in sense_count.items():
                if count > max_count:
                    remove = False
                    max_count = count
                    sense_idx = sense_index
                elif count == max_count:
                    remove = True

            if not remove:
                final_word_sense_count[word_key] = sense_idx

        results = []
        for i in range(len(self.document)):
            key = f"{self.document.words[i]}||{self.document.words_pos[i]}"
            if key in final_word_sense_count:
                results.append((i, final_word_sense_count[key]))
            else:
                results.append(senses[i])

        return results

    def convertFinalSynsets(self, final_senses: List[Tuple[int, int]]) -> List[Optional[Synset]]:
        synsets = []
        for word_index, final_sense in enumerate(final_senses):
            synsets.append(self.get_synset(final_sense))

        return synsets

    def get_synset(self, final_sense: Tuple[int, int]) -> Optional[Synset]:
        if final_sense is None:
            return None

        word_index, sense_index = final_sense
        synsets = SynsetUtils.get_wordnet_synsets(self.document.words[word_index],
                                                  pos=self.document.words_pos[word_index],
                                                  lemma=self.document.words_lemma[word_index])
        if len(synsets) == 0:
            return None
        else:
            return synsets[sense_index]
