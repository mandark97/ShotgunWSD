from typing import List
from xml.dom import minidom

from parsed_document import Document


class Parser(object):
    def __init__(self, filename: str):
        self.filename = filename
        super().__init__()

    def run(self) -> List[Document]:
        xmldoc = minidom.parse(self.filename)
        texts = xmldoc.getElementsByTagName("text")

        documents = []
        for text in texts:
            text_id = text.getAttribute("id")
            words = []
            words_pos = []
            words_lemma = []
            words_id = []
            for sentence in text.childNodes:
                for word in sentence.childNodes:
                    if word.nodeName not in ["wf", "instance"]:
                        continue
                    words.append(word.childNodes[0].nodeValue)
                    words_pos.append(word.getAttribute("pos"))
                    words_lemma.append(word.getAttribute("lemma"))
                    words_id.append(word.getAttribute("id"))

            documents.append(Document(text_id, words, words_pos, words_lemma, words_id))

        return documents
