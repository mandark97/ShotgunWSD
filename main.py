import logging

from lesk import Lesk
from operations import AddOperation, LogOperation, SumSquaredOperation
from parser import Parser
from shotgun_wsd import ShotgunWSD
from synset_utils import SynsetUtils

documents = Parser().run()
logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')

document = documents[0]
synset_relatedness = Lesk()
SynsetUtils.configuration_operation = AddOperation()
SynsetUtils.synset_relatedness = synset_relatedness
shotgun_wsd = ShotgunWSD(document=document, window_size=2, number_configs=2, synset_relatedness=synset_relatedness,
                         min_synset_collision=2, max_synset_collision=4, number_of_votes=4)
final_senses = shotgun_wsd.run()
with open("results.txt", "w") as file:
    for t in final_senses:
        file.write(str(t))
        file.write("\n")