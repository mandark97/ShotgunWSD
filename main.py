from typing import List, Dict

from local_wsd import LocalWSD
from parsed_document import Document
from synset_relatedness import SynsetRelatedness
from window_config import WindowConfiguration


class ShotgunWSD(object):
    def __init__(self, document: Document, window_size: int, number_configs: int, synset_relatedness: SynsetRelatedness,
                 min_synset_collision: int, max_synset_collision: int):
        self.document = document
        self.window_size = window_size
        self.number_configs = number_configs
        self.synset_relatedness = synset_relatedness
        self.min_synset_collision = min_synset_collision
        self.max_synset_collision = max_synset_collision

        super().__init__()

    def run(self):
        documentWindowSolutions: Dict[int, List[WindowConfiguration]] = self.compute_windows()
        self.merge_window_solutions(documentWindowSolutions)
        senseVotes = self.vote_senses(documentWindowSolutions)
        senses = self.selectSenses(documentWindowSolutions, senseVotes)
        finalSenses = self.detectMostUsedSenses(senses)
        convertedSynsets = self.convertFinalSynsets(finalSenses)

        return convertedSynsets

    def compute_windows(self) -> Dict[int, List[WindowConfiguration]]:
        document_window_solutions = {}

        for word_index in range(0, len(self.document.words) - self.window_size):
            window_words = self.document.words[word_index:word_index + self.window_size]
            window_words_pos = self.document.words_pos[word_index:word_index + self.window_size]

            # TODO some max synset combination black magic
            local_wsd = LocalWSD(word_index, window_words, window_words_pos,
                                 self.number_configs, self.synset_relatedness)
            local_wsd.run()

            document_window_solutions[word_index] = local_wsd.windows_solutions

        return document_window_solutions

    def merge_window_solutions(self, document_window_solutions):
        merged_windows = None

        for synset_collisions in reversed(range(self.min_synset_collision, self.max_synset_collision)):
            merged_windows = self.merge_windows(document_window_solutions, synset_collisions)

        return merged_windows

    # works?
    def merge_windows(self, document_window_solutions: Dict[int, List[WindowConfiguration]],
                      synset_collisions: int) -> Dict[int, List[WindowConfiguration]]:
        for l in range(len(self.document.words)):
            if l in document_window_solutions:
                config_list1 = document_window_solutions[l]
                for window1 in config_list1:
                    for j in range(len(window1.synset_indexes) - synset_collisions):
                        if j + l + 1 in document_window_solutions:
                            config_list2 = document_window_solutions[j + l + 1]

                            for window2 in config_list2:
                                # collided = False

                                if WindowConfiguration.has_collisions(window1, window2, j + 1, synset_collisions):
                                    merged_window = WindowConfiguration.merge(window1, window2, j + 1)
                                    if merged_window != None:
                                        # collided = True
                                        config_list1.append(merged_window)
                                    config_list2.remove(window2)

        return document_window_solutions

    def vote_senses(self, documentWindowSolutions):
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
