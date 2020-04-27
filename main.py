import logging

from relatedness.lesk import Lesk
from relatedness.word_embedding import WordEmbeddingRelatedness
from operations import AddOperation
from parser import Parser
from result_writer import ResultWriter
from shotgun_wsd import ShotgunWSD
from synset_utils import SynsetUtils

documents = Parser().run()
logging.basicConfig(level=logging.DEBUG, format='%(message)s')

document = documents[0]
synset_relatedness = Lesk()
# synset_relatedness = WordEmbeddingRelatedness()
# synset_relatedness = WordEmbeddingRelatedness('median')
SynsetUtils.configuration_operation = AddOperation()
SynsetUtils.synset_relatedness = synset_relatedness
shotgun_wsd = ShotgunWSD(document=document, window_size=2, number_configs=2, synset_relatedness=synset_relatedness,
                         min_synset_collision=2, max_synset_collision=4, number_of_votes=4)
final_senses = shotgun_wsd.run()

ResultWriter(document, final_senses).write("results.txt", mode=ResultWriter.SCORE)
