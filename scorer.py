from collections import defaultdict
from typing import Dict, Set


class Scorer(object):

    def __init__(self, answers_path: str, results_path: str = None, results: Dict[str, Set[str]] = None):
        if results is not None:
            self.results = results
        else:
            self.results = self._parse_file(results_path)

        self.answers = self._parse_file(answers_path)

    def score(self) -> Dict[str, float]:
        correct, wrong, total = 0, 0, 0

        for pos in self.answers:
            if pos not in self.results:
                continue

            if self.results[pos].issubset(self.answers[pos]):
                correct += 1
            else:
                wrong += 1
            total += 1

        precision = correct / (correct + wrong)
        recall = correct / total
        f1 = 0
        if (precision + recall) != 0:
            f1 += (precision * recall * 2)
            f1 /= (precision + recall)

        return {
            'correct': correct,
            'wrong': wrong,
            'total': total,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def _parse_file(self, path: str) -> Dict[str, Set[str]]:
        word_dict = defaultdict(set)

        with open(path, 'r') as file:
            for line in file:
                pos, *senses = line.split()

                if senses is None:
                    continue

                for sense in senses:
                    word_dict[pos].add(sense)

        return word_dict
