import logging

from relatedness.lesk import Lesk
from relatedness.word_embedding import WordEmbeddingRelatedness
from operations import AddOperation
from parser import Parser
from result_writer import ResultWriter
from scorer import Scorer
from shotgun_wsd import ShotgunWSD
from synset_utils import SynsetUtils


DOCUMENTS_PATH = "datasets/semeval2013/semeval2013.data.xml"
ANSWERS_PATH = "datasets/semeval2013/semeval2013.gold.key.txt"
RESULTS_PATH = "results.txt"

documents = Parser(filename=DOCUMENTS_PATH).run()
logging.basicConfig(level=logging.DEBUG, format='%(message)s')

document = documents[0]
synset_relatedness = Lesk()
# synset_relatedness = WordEmbeddingRelatedness()
# synset_relatedness = WordEmbeddingRelatedness('median')
SynsetUtils.configuration_operation = AddOperation()
SynsetUtils.synset_relatedness = synset_relatedness
shotgun_wsd = ShotgunWSD(document=document, window_size=8, number_configs=15, synset_relatedness=synset_relatedness,
                         min_synset_collision=1, max_synset_collision=4, number_of_votes=15)
final_senses = shotgun_wsd.run()

ResultWriter(document, final_senses).write(RESULTS_PATH, mode=ResultWriter.SCORE)

scorer = Scorer(RESULTS_PATH, ANSWERS_PATH)
scorer_results = scorer.score()
print(scorer_results)