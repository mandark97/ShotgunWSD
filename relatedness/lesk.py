import nltk
from relatedness.synset_relatedness import SynsetRelatedness
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.corpus.reader.wordnet import Synset
from nltk.stem.snowball import SnowballStemmer


def stringify(f):
    """
    Wraps around any function f(s: Synset) -> list[Synset]:
    and returns the concatenated glosses of the list elements
    :returns str
    """

    def inner(synset: Synset) -> str:
        synsets = f(synset)
        return " ".join([s.definition() for s in synsets])

    return inner


@stringify
def get_meronyms(synset: Synset) -> str:
    return synset.part_meronyms() + synset.substance_meronyms()


@stringify
def get_hyponyms(synset: Synset) -> str:
    return synset.hyponyms()


@stringify
def get_hypernyms(synset: Synset) -> str:
    return synset.hypernyms()


@stringify
def get_attributes(synset: Synset) -> str:
    return synset.attributes()


@stringify
def get_also_sees(synset: Synset) -> str:
    return synset.also_sees()


class Lesk(SynsetRelatedness):
    relations = {
        'hyper': get_meronyms,
        'hypo': get_hyponyms,
        'mero': get_meronyms,
        'ex': lambda s: " ".join(s.examples()),
        'gloss': lambda s: s.definition(),
        'attr': get_attributes,
        'also': get_also_sees,
    }

    def __init__(self):
        super().__init__()

    @staticmethod
    def score_relation_pairs(synset1: Synset, synset2: Synset, relations: list) -> float:
        score = 0
        pool = {
            synset1: {},
            synset2: {}
        }

        def get_rel(synset: Synset, rel: str):
            if rel not in Lesk.relations:
                raise Exception(f"Relation '{rel}' not defined")
            if rel not in pool[synset]:
                pool[synset][rel] = Lesk.relations[rel](synset)
            return pool[synset][rel]

        relations += [(r2, r1) for (r1, r2) in relations if (r2, r1) not in relations]
        for r1, r2 in relations:
            synset1_rel1 = get_rel(synset1, r1)
            synset2_rel2 = get_rel(synset2, r2)
            score += Lesk.get_score(synset1_rel1, synset2_rel2)
        return score

    @staticmethod
    def compute_noun_similarity(synset1: Synset, synset2: Synset) -> float:
        return Lesk.score_relation_pairs(synset1, synset2, [
            ('hypo', 'mero'),
            ('hypo', 'hypo'),
            ('gloss', 'mero'),
            ('gloss', 'gloss'),
            ('ex', 'mero'),
        ])

    @staticmethod
    def compute_adjective_similarity(synset1: Synset, synset2: Synset) -> float:
        return Lesk.score_relation_pairs(synset1, synset2, [
            ('also', 'gloss'),
            ('attr', 'gloss'),
            ('gloss', 'gloss'),
            ('ex', 'gloss'),
            ('gloss', 'hyper'),
        ])

    @staticmethod
    def compute_verb_similarity(synset1: Synset, synset2: Synset) -> float:
        return Lesk.score_relation_pairs(synset1, synset2, [
            ('ex', 'ex'),
            ('ex', 'hyper'),
            ('hypo', 'hypo'),
            ('gloss', 'hypo'),
            ('ex', 'gloss')
        ])

    @staticmethod
    def compute_simple_similarity(synset1: Synset, synset2: Synset) -> float:
        gloss1 = Lesk.relations['gloss'](synset1) + Lesk.relations['ex'](synset1)
        gloss2 = Lesk.relations['gloss'](synset2) + Lesk.relations['ex'](synset2)
        return Lesk.get_score(gloss1, gloss2)

    @staticmethod
    def clean(s: str):
        # Tokenize
        s = nltk.word_tokenize(s)

        # Stopwords and noise removal
        sw = set(stopwords.words('english'))
        s = [w for w in s if w not in sw and "'" not in w]

        # Stemming
        stemmer = SnowballStemmer('english')
        s = [stemmer.stem(w) for w in s]
        return s

    @staticmethod
    def get_score(gloss1: str, gloss2: str):
        gloss1 = Lesk.clean(gloss1)
        gloss2 = Lesk.clean(gloss2)
        score = 0
        has_diffs = True
        while has_diffs:
            max_over_size = 0
            max_over_i = -1
            max_over_j = -1
            for i, w1 in enumerate(gloss1):
                for j, w2 in enumerate(gloss2):
                    # overlap, measure it
                    if w1 == w2:
                        over_size = 1
                        oi = i + 1
                        oj = j + 1
                        while oi < len(gloss1) and oj < len(gloss2) and gloss1[oi] == gloss2[oj]:
                            oi += 1
                            oj += 1
                            over_size += 1
                        if over_size > max_over_size:
                            max_over_size = over_size
                            max_over_i = i
                            max_over_j = j
            if max_over_size == 0:
                has_diffs = False
            else:
                score += max_over_size ** 2
                del gloss1[max_over_i:max_over_i + max_over_size]
                del gloss2[max_over_j:max_over_j + max_over_size]
        return score

    def compute_similarity(self, word1: str, synset1: Synset, word2: str, synset2: Synset) -> float:
        if synset1 is None or synset2 is None:
            return 0.

        pos1 = synset1.pos()
        pos2 = synset2.pos()
        if pos1 == pos2:
            if pos1 == wn.NOUN:
                return self.compute_noun_similarity(synset1, synset2)
            elif pos1 == wn.VERB:
                return self.compute_verb_similarity(synset1, synset2)
            elif pos1 == wn.ADJ:
                return self.compute_adjective_similarity(synset1, synset2)
        return self.compute_simple_similarity(synset1, synset2)
