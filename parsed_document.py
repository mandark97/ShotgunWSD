class Document(object):
    def __init__(self, id, words, words_pos, words_lema):
        self.id = id
        self.words = words
        self.words_pos = words_pos
        self.words_lema = words_lema
        super().__init__()
