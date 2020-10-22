from agent.planning.planner import Planner

"""
Implements the breadth first search algorithm.
"""

from collections import deque
import logging
from representation import searchspace


class ForwardSearch(Planner):
    """
    Regular BFS starting from the initial state and capturing visited nodes

    """

    def __init__(self):
        super().__init__()

    def _search(self, planning_task, closed_set=None):
        """
        Searches for a plan on the given task using breadth first search and
        duplicate detection.
        @param planning_task: The planning task to solve.
        @return: The solution as a list of operators or None if the task is
        unsolvable.

        Frontier: The states that are the end of a particular search expedition.
          no successor state which is still not closed
        Closed: The states that the BFS has visited.
        """
        # counts the number of loops (only for printing)
        iteration = 0
        # fifo-queue storing the nodes which are next to explore
        queue = deque()
        root_node = searchspace.make_root_node(planning_task.initial_state)
        queue.append(root_node)
        frontier = []
        # set storing the explored nodes, used for duplicate detection
        if closed_set is None:
            closed = set()
        else:
            closed = set(closed_set)
        closed.add(tuple(planning_task.initial_state))
        while queue:
            iteration += 1
            logging.debug(
                "breadth_first_search: Iteration %d, #unexplored=%d"
                % (iteration, len(queue))
            )
            # get the next node to explore
            node = queue.popleft()
            if planning_task.goal_reached(node.state):
                logging.info("Goal reached. Start extraction of solution.")
                logging.info("%d Nodes expanded" % iteration)
                return node.extract_solution(), frontier, closed
            is_frontier = True
            for operator, successor_state in planning_task.get_successor_states(node.state):
                # duplicate detection
                if successor_state not in closed:
                    is_frontier = False
                    queue.append(
                        searchspace.make_child_node(node, operator, successor_state)
                    )
                    # remember the successor state
                    closed.add(successor_state)
            if is_frontier:
                frontier.append(node)
        logging.info("No operators left. Task unsolvable.")
        logging.info("%d Nodes expanded" % iteration)
        return None, frontier, closed
