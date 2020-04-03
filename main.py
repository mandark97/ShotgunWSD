from typing import List, Tuple, Union

from nltk.corpus import wordnet as wn
from nltk.corpus.reader import Synset

from parsed_document import Document
from pos_utils import get_pos
from synset_relatedness import SynsetRelatedness


class ShotgunWSD(object):
    def __init__(self, document: Document, window_size: int, number_configs: int):
        self.document = document
        self.window_size = window_size
        self.number_configs = number_configs
        self.synset_relatedness = None
        super().__init__()

    def run(self):
        documentWindowSolutions = self.compute_windows()
        self.mergeWindowSolutions(documentWindowSolutions)
        senseVotes = self.voteSenses(documentWindowSolutions)
        senses = self.selectSenses(documentWindowSolutions, senseVotes)
        finalSenses = self.detectMostUsedSenses(senses)
        convertedSynsets = self.convertFinalSynsets(finalSenses)

        return convertedSynsets

    def compute_windows(self):
        for word_index in range(0, len(self.document.words) - self.window_size):
            window_words = self.document.words[word_index:word_index + self.window_size]
            window_words_pos = self.document.words_pos[word_index:word_index + self.window_size]

            # some max synset combination black magic
            LocalWSD(word_index, window_words, window_words_pos,
                     self.number_configs, self.synset_relatedness)

    def mergeWindowSolutions(self, documentWindowSolutions):
        pass

    def voteSenses(self, documentWindowSolutions):
        pass

    def selectSenses(self, documentWindowSolutions, senseVotes):
        pass

    def detectMostUsedSenses(self, senses):
        pass

    def convertFinalSynsets(self, senses):
        pass


# documents = Parser().run()

# for document in documents:
#     ShotgunWSD(document=document, window_size=2, number_configs=2).run()


class LocalWSD(object):
    def __init__(self, word_index: int, window_words: List[str], window_words_pos: List[str], number_configs: int,
                 synset_relatedness: SynsetRelatedness):
        self.word_index = word_index
        self.window_words = window_words
        self.window_words_pos = window_words_pos
        self.number_configs = number_configs
        self.synset_relatedness = synset_relatedness

        super().__init__()

    def run(self):
        word_synsets = self.build_window_synsets_array()
        similarity_matrix = self.compute_relatedness(word_synsets)

        self.generate_synset_combinations(word_synsets, similarity_matrix)

    def build_window_synsets_array(self) -> List[Union[Tuple[str, None], Tuple[str, Synset]]]:
        word_synsets = []
        for index, word in enumerate(self.window_words):
            synsets = wn.synsets(word, pos=get_pos(self.window_words_pos[index]))
            if len(synsets) == 0:
                word_synsets.append((word, None))
            else:
                for synset in synsets:
                    word_synsets.append((word, synset))

        return word_synsets

    def compute_relatedness(self, word_synsets: List[Tuple[str, Synset]]) -> List[List[float]]:
        similarity_matrix = [[0.0 for _ in range(len(word_synsets))] for _ in range(len(word_synsets))]

        for i, (word1, synset1) in enumerate(word_synsets):
            for j, (word2, synset2) in enumerate(word_synsets[i:], start=i):
                sim = self.synset_relatedness.compute_similarity(word1, synset1, word2, synset2)
                similarity_matrix[i][j] = sim
                similarity_matrix[j][i] = sim

        return similarity_matrix

    # get the best configurations
    def generate_synset_combinations(self, word_synsets, similarity_matrix):
        pass
