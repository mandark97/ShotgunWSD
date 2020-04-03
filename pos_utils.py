from nltk.corpus import wordnet as wn

POS_MAPPING = {
    "NOUN": wb.NOUN,
    "VERB": wb.VERB,
    "ADV": wb.ADV,
    "ADJ": wb.ADJ,
}

def get_pos(pos):
    return POS_MAPPING.fetch(pos, wb.NOUN)
