from abc import ABC, abstractmethod
import math

class ConfigurationOperation(ABC):
    def __init__(self):
        self.initial_score = 0

    @abstractmethod
    def apply_operation(self, accumulator: float, value: float) -> float:
        pass


class AddOperation(ConfigurationOperation):
    def __init__(self):
        super(AddOperation, self).__init__()

    def apply_operation(self, accumulator: float, value: float) -> float:
        return accumulator + value


class LogOperation(ConfigurationOperation):
    def __init__(self):
        super(LogOperation, self).__init__()

    def apply_operation(self, accumulator: float, value: float) -> float:
        return accumulator + math.log2(2 + value)


class SumSquaredOperation(ConfigurationOperation):
    def __init__(self):
        super(SumSquaredOperation, self).__init__()

    def apply_operation(self, accumulator: float, value: float) -> float:
        return accumulator + value**2