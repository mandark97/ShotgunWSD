from typing import List, Optional, TextIO, Set, Dict

from nltk.corpus.reader import Synset

from parsed_document import Document
from synset_utils import SynsetUtils


class ResultWriter(object):
    HUMAN = "human"
    SCORE = "score"

    def __init__(self, document: Document, results: List[Optional[Synset]]):
        self.document = document
        self.results = results

    def to_dict(self) -> Dict[str, Set[str]]:
        results = {}
        for word_index, synset in enumerate(self.results):
            id = self.document.words_id[word_index]
            if id == "":
                continue

            if synset is None:
                sense = ""
            else:
                sense = SynsetUtils.sense_key(synset, self.document.words_lemma[word_index])

            results[id] = {sense}

        return results

    def write(self, output_path: str, mode: str = "human"):
        with open(output_path, "w") as file_writer:
            if mode == self.HUMAN:
                self._human_write(file_writer)
            elif mode == self.SCORE:
                self._score_write(file_writer)

    def _human_write(self, file_writer: TextIO):
        for word_index, synset in enumerate(self.results):
            if synset is None:
                sense = ""
            else:
                sense = synset.definition()

            file_writer.write(
                f"Word: {self.document.words[word_index]}, Lemma: {self.document.words_lemma[word_index]}, Definition: {sense}\n")

    def _score_write(self, file_writer: TextIO):
        for word_index, synset in enumerate(self.results):
            id = self.document.words_id[word_index]
            if id == "":
                continue

            if synset is None:
                sense = ""
            else:
                sense = SynsetUtils.sense_key(synset, self.document.words_lemma[word_index])
            file_writer.write(f"{id} {sense}\n")
