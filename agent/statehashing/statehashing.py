from abc import ABC, abstractmethod


class StateHasher(ABC):
    def __init__(self):
        pass

    def hash(self, obs):
        return self._hash(obs)

    @abstractmethod
    def _hash(self, obs):
        pass
