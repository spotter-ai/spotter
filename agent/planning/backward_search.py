from agent.planning.planner import Planner

"""
Implements the breadth first search algorithm.
"""

from collections import deque
import logging
from representation import searchspace


class BackwardSearch(Planner):
    """
    Regression BFS starting from the goal condition and capturing visited nodes
    node.state is not a state, but a condition

    """

    def __init__(self):
        pass

    def _search(self, planning_task):
        """
        Note here: we use the SearchNode class which has "state" but instead, we interpret here that
        it is actually just a condition.
        """
        import time
        start = time.time()
        # counts the number of loops (only for printing)
        iteration = 0
        # fifo-queue storing the nodes which are next to explore
        queue = deque()
        root_node = searchspace.make_root_node(planning_task.goals)
        queue.append(root_node)
        frontier = []
        # set storing the explored nodes, used for duplicate detection
        closed = {planning_task.goals}
        while queue:
            iteration += 1
            logging.debug(
                "breadth_first_search: Iteration %d, #unexplored=%d"
                % (iteration, len(queue))
            )
            # get the next node to explore
            node = queue.popleft()
            if planning_task.start_reached(node.state):
                logging.info("Goal reached. Start extraction of solution.")
                logging.info("%d Nodes expanded" % iteration)
                return node.extract_regression(), frontier, closed
            is_frontier = True
            for operator,regressor_condition in planning_task.get_regressor_conditions(node.state):
                # duplicate detection
                duplicate = False
                for match_conditions in closed:
                    if regressor_condition[0] >= match_conditions[0] and regressor_condition[1] >= match_conditions[1]:
                        duplicate = True
                if not duplicate:
                    is_frontier = False
                    queue.append(
                        searchspace.make_child_node(node, operator, regressor_condition)
                    )
                    # remember the successor state
                    closed.add(regressor_condition)
            if is_frontier:
                frontier.append(node)
        minimized = set(closed)
        for sub in closed:
            supersets = [sup for sup in minimized if sub != sup and sub[0] <= sup[0] and sub[1] <= sup[1]]
            for sup in supersets:
                minimized.remove(sup)
        closed = minimized
        end = time.time()
        logging.info("Elapsed time: " + str(end - start))
        print("Elapsed time: " + str(end - start))
        logging.info("No operators left. Task unsolvable.")
        logging.info("%d Nodes expanded" % iteration)
        return None, frontier, closed
