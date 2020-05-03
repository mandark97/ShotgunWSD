from typing import Dict, Set


class Scorer(object):

    def __init__(self, results_path: str, answers_path: str):
        self.results = self._parse_file(results_path)
        self.answers = self._parse_file(answers_path)

    def score(self) -> Dict[str, float]:
        total = correct = wrong = 0

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
        word_dict = dict()

        file = open(path, 'r')
        for line in file:
            elements = line.split()

            if len(elements) < 2:
                continue

            pos = elements[0]
            if pos not in word_dict:
                word_dict[pos] = set()

            for sense in elements[1:]:
                word_dict[pos].add(sense)

        return word_dict
