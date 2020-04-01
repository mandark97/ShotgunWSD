from nltk.corpus import wordnet as wn

from parsed_document import Document
from parsers.senseval import Parser


class ShotgunWSD(object):
    def __init__(self, document: Document, window_size: int, number_configs: int):
        self.document = document
        self.window_size = window_size
        self.number_configs = number_configs
        super().__init__()

    def run(self):
        senses = []

        for word in self.document.words:
            senses.append(wn.synsets(word))


documents = Parser().run()

for document in documents:
    ShotgunWSD(document=document, window_size=2, number_configs=2).run()
