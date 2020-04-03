from nltk.corpus import wordnet as wn

POS_MAPPING = {
    "NOUN": wn.NOUN,
    "VERB": wn.VERB,
    "ADV": wn.ADV,
    "ADJ": wn.ADJ,
}


def get_pos(pos):
    return POS_MAPPING.fetch(pos, wn.NOUN)
