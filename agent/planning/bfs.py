from agent.planning.planner import Planner

"""
Implements the breadth first search algorithm.
"""

from collections import deque
import logging
from representation import searchspace


class BFS(Planner):
    """
    Breadth-First Search implementation for planner
    """

    def __init__(self):
        pass

    def _search(self, planning_task):
        """
        Searches for a plan on the given task using breadth first search and
        duplicate detection.
        @param planning_task: The planning task to solve.
        @return: The solution as a list of operators or None if the task is
        unsolvable.
        """
        # counts the number of loops (only for printing)
        iteration = 0
        # fifo-queue storing the nodes which are next to explore
        queue = deque()
        queue.append(searchspace.make_root_node(planning_task.initial_state))
        # set storing the explored nodes, used for duplicate detection
        closed = {planning_task.initial_state}
        while queue:
            iteration += 1
            logging.debug(
                "breadth_first_search: Iteration %d, #unexplored=%d"
                % (iteration, len(queue))
            )
            # get the next node to explore
            node = queue.popleft()
            # exploring the node or if it is a goal node extracting the plan
            if planning_task.goal_reached(node.state):
                logging.info("Goal reached. Start extraction of solution.")
                logging.info("%d Nodes expanded" % iteration)
                return node.extract_solution()
            for operator, successor_state in planning_task.get_successor_states(node.state):
                # duplicate detection
                if successor_state not in closed:
                    queue.append(
                        searchspace.make_child_node(node, operator, successor_state)
                    )
                    # remember the successor state
                    closed.add(successor_state)
        logging.info("No operators left. Task unsolvable.")
        logging.info("%d Nodes expanded" % iteration)
        return None
