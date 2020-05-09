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
logging.basicConfig(level=logging.INFO,
                    format='%(message)s', filename="run.log")

# document = documents[0]
synset_relatedness = Lesk()
# synset_relatedness = WordEmbeddingRelatedness()
# synset_relatedness = WordEmbeddingRelatedness('median')
SynsetUtils.configuration_operation = AddOperation()
SynsetUtils.synset_relatedness = synset_relatedness

results_dict = {}
for document in documents:
    shotgun_wsd = ShotgunWSD(document=document, window_size=2, number_configs=4, synset_relatedness=synset_relatedness,
                             min_synset_collision=1, max_synset_collision=4, number_of_votes=10)
    final_senses = shotgun_wsd.run()

    result_writer = ResultWriter(document, final_senses)
    results_dict.update(result_writer.to_dict())

# result_writer.write(RESULTS_PATH, mode=ResultWriter.SCORE)
scorer = Scorer(results=results_dict, answers_path=ANSWERS_PATH)
scorer_results = scorer.score()
print(scorer_results)
