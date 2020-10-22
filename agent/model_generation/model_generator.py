from abc import ABC

class ModelGenerator(ABC):
    def __init__(self):
        pass

    def _generate(self, forward, backward, execution):
        """
        Implement generator
        """

    def generate(self, forward, backward, execution):
        return self._generate(forward, backward, execution)