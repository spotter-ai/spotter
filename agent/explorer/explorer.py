from abc import abstractmethod, ABC


class Explorer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def _explore(self):
        """
        implement
        """

    def explore(self):
        return self._explore()