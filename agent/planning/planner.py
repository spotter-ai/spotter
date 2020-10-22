from abc import ABC, abstractmethod
from typing import List

from representation.task import Operator


class Planner(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def _search(self, planning_task, closed_set=None) -> List[Operator]:
        """
        This needs to be implemented by the underlying planner
        :return:
        """

    def search(self, planning_task, closed_set=None) -> List[Operator]:
        return self._search(planning_task, closed_set)

