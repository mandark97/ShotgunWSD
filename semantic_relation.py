from nltk.corpus.reader.wordnet import Synset

from typing import List


class SemanticRelation(object):

    @staticmethod
    def get_hyponyms(synset: Synset) -> List:
        return synset.hyponyms()

    @staticmethod
    def get_hypernyms(synset: Synset) -> List:
        return synset.hypernyms()

    @staticmethod
    def get_meronyms(synset: Synset) -> List:
        return synset.part_meronyms() + synset.substance_meronyms()

    @staticmethod
    def get_antonyms(synset: Synset) -> List:
        return [antonym
                for lemma in synset.lemmas()
                for antonym in lemma.antonyms()]

    @staticmethod
    def get_pertainyms(synset: Synset) -> List:
        return [pertainym
                for lemma in synset.lemmas()
                for pertainym in lemma.pertainyms()]

    @staticmethod
    def get_entailments(synset: Synset) -> List:
        return synset.entailments()

    @staticmethod
    def get_attributes(synset: Synset) -> List:
        return synset.attributes()

    @staticmethod
    def get_sense_keys(synset: Synset) -> List:
        return synset.lemma_names()

    @staticmethod
    def get_topics(synset: Synset) -> List:
        return synset.topic_domains() + synset.in_topic_domains()
