from abc import ABC, abstractmethod
from typing import Type

from agent.brain import Brain


class Mode(ABC):
    """
    Agent's mode of operation
    This is essentially a state of the agent, not a state in the world
    """

    @abstractmethod
    def __init__(self, brain: Brain):
        self.brain = brain
        #print("Processing current mode: ", str(self))

    @abstractmethod
    def _run(self):
        """
        Main run loop and needs to be implemented
        :return:
        """
        pass

    @abstractmethod
    def _next(self):
        """
        Logic for transitioning to next mode. Needs to be implemented
        :return:
        """
        return None

    def run(self):
        self._run()
        return None

    def next(self):
        mode = self._next()
        return mode

