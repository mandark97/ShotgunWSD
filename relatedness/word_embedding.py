import nltk
import gensim.downloader as api
import numpy as np

from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset
from nltk.stem.snowball import SnowballStemmer
from os import path
from scipy.stats.mstats import gmean
from scipy.spatial.distance import cosine

from relatedness.synset_relatedness import SynsetRelatedness
from semantic_relation import SemanticRelation

from typing import List


class WordEmbeddingRelatedness(SynsetRelatedness):

    def __init__(self, computation_sense: str = 'average'):
        self.model_path = api.BASE_DIR + \
            '/word2vec-google-news-300/word2vec-google-news-300.gz'

        if not path.exists(self.model_path):
            self.model_path = api.load("word2vec-google-news-300",
                                       return_path=True)

        self.word_vec_model = KeyedVectors.load_word2vec_format(
                                    self.model_path,
                                    binary=True)

        self.extended_vocab = dict()

        if computation_sense == 'median':
            self.compute_sense = self.median_computation_sense
        else:
            self.compute_sense = self.average_computation_sense

    def compute_similarity(self,
                           word1: str, synset1: Synset,
                           word2: str, synset2: Synset) -> float:
        sense_embedding1 = self.get_sense_embedding(synset1, word1)
        sense_embedding2 = self.get_sense_embedding(synset2, word2)

        return cosine(sense_embedding1, sense_embedding2)

    def get_sense_embedding(self, synset: Synset, word: str) -> np.ndarray:
        if word in self.word_vec_model:
            return self.word_vec_model.word_vec(word)

        if word in self.extended_vocab:
            return self.extended_vocab[word]

        words = self.get_sense_bag(synset)

        sense_embeddings = []
        for w in words:
            if w is not None and w in self.word_vec_model:
                temp_embedding = self.word_vec_model.word_vec(w)
                sense_embeddings.append(temp_embedding[:])

        sense_embedding = self.compute_sense(sense_embeddings)
        self.extended_vocab[word] = sense_embedding[:]

        return sense_embedding

    def get_sense_bag(self, synset: Synset) -> List:
        if synset is None:
            return []

        pos = synset.pos()
        if pos == wn.NOUN:
            return self.get_noun_sense_bag(synset)
        elif pos == wn.VERB:
            return self.get_verb_sense_bag(synset)
        elif pos == wn.ADJ:
            return self.get_adj_sense_bag(synset)
        elif pos == wn.ADV:
            return self.get_adv_sense_bag(synset)

        return[]

    def get_noun_sense_bag(self, synset: Synset) -> List:
        sense_bag = self.get_synset_bag(synset)
        hyponyms = SemanticRelation.get_hyponyms(synset)
        meronyms = SemanticRelation.get_meronyms(synset)

        for hyponym in hyponyms:
            sense_bag += self.get_synset_bag(hyponym)

        for meronym in meronyms:
            sense_bag += self.get_synset_bag(meronym)

        return self.get_word_set_from_text(sense_bag)

    def get_verb_sense_bag(self, synset: Synset) -> List:
        sense_bag = self.get_synset_bag(synset)
        entailments = SemanticRelation.get_entailments(synset)
        hypernyms = SemanticRelation.get_hypernyms(synset)

        for entailment in entailments:
            sense_bag += self.get_synset_bag(entailment)

        for hypernym in hypernyms:
            sense_bag += self.get_synset_bag(hypernym)

        return self.get_word_set_from_text(sense_bag)

    def get_adj_sense_bag(self, synset: Synset) -> List:
        sense_bag = self.get_synset_bag(synset)
        attributes = SemanticRelation.get_attributes(synset)
        antonyms = SemanticRelation.get_antonyms(synset)
        pertainyms = SemanticRelation.get_pertainyms(synset)

        for attribute in attributes:
            sense_bag += self.get_synset_bag(attribute)

        for antonym in antonyms:
            sense_bag += f' {antonym.name()} '

        for pertainym in pertainyms:
            sense_bag += f' {pertainym.name()} '

        return self.get_word_set_from_text(sense_bag)

    def get_adv_sense_bag(self, synset: Synset) -> List:
        sense_bag = self.get_synset_bag(synset)
        topics = SemanticRelation.get_topics(synset)
        antonyms = SemanticRelation.get_antonyms(synset)
        pertainyms = SemanticRelation.get_pertainyms(synset)

        for topic in topics:
            sense_bag += self.get_synset_bag(topic)

        for antonym in antonyms:
            sense_bag += f' {antonym.name()} '

        for pertainym in pertainyms:
            sense_bag += f' {pertainym.name()} '

        return self.get_word_set_from_text(sense_bag)

    def get_synset_bag(self, synset: Synset) -> str:
        sense_bag = synset.definition()
        sense_examples = synset.examples()
        sense_keys = SemanticRelation.get_sense_keys(synset)

        for sense_example in sense_examples:
            sense_bag += f' {sense_example} '

        for sense_key in sense_keys:
            sense_bag += f' {sense_key} '

        return f' {sense_bag} '

    def average_computation_sense(self, sense_embeddings: List) -> List:
        if not sense_embeddings:
            return np.zeros(300)
        return np.mean(sense_embeddings, axis=0)

    def median_computation_sense(self, sense_embeddings: List) -> List:
        if not sense_embeddings:
            return np.zeros(300)
        return gmean(sense_embeddings)

    def get_word_set_from_text(self, text: str) -> List:
        stop_words = set(stopwords.words('english'))
        stemmer = SnowballStemmer('english')
        words = nltk.word_tokenize(text)

        words = [word for word in words
                 if word not in stop_words and "'" not in word]
        words = [stemmer.stem(word) for word in words]
        return list(set(words))
