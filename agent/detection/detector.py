from abc import ABC, abstractmethod


class Detector(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def _interpret(self, state):
        """"
        Needs to be implemented
        Given a state, the detector returns the fluent state
        """

    @abstractmethod
    def _all_fluents(self):
        """
        Needs implementation
        Construct a list of all fluents which could possibly be true in a given environment.
        """

    @abstractmethod
    def _detect(self):
        """
        Needs implementation
        Queries env for last observation and then interprets it to return fluent state
        """

    @abstractmethod
    def _get_objects(self, **kwargs):
        """
        Returns current object types, and what object are included in the environment
        Map: {type, list of objects}
        E.g., {"block":["block_blue", "block_red"], "cup":["cup"]}
        """

    @abstractmethod
    def _check_formula(self, formula, state):
        """
        Needs to be implemented
        Check if a conjunctive formula holds in a state
        """

    def all_fluents(self):
        return self._all_fluents()

    def detect(self):
        return self._detect()

    def interpret(self, state):
        return self._interpret(state)

    def get_objects(self, **kwargs):
        return self._get_objects(kwargs)

    def check_formula(self, formula, state):
        return self._check_formula(formula, state)
