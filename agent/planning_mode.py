import logging

from agent.execution_mode import ExecutionMode
from agent.exploration_mode import ExplorationMode
from agent.planning.forward_search import ForwardSearch
from agent.mode import Mode
from representation.task import state_subsumes

from helper import get_root_dir
logging.basicConfig(filename=get_root_dir() + "/logs/console.log",
                    format="%(name)s — %(levelname)s — %(message)s",
                    filemode='w')
log = logging.getLogger("planning_mode")
log.setLevel(logging.DEBUG)


class PlanningMode(Mode):
    """
    Planning mode
    """

    def __init__(self, brain, planner="bfs"):
        super().__init__(brain)
        self.name = "Planning Mode"
        self.planner = planner
        self.plan_exists = False

    def _run(self):
        # TODO implement plan caching and unplannability caching in the planning mode so we don't have to
        #  replan every time.
        if self.planner == "bfs":
            # bfs = BFS()
            forward_search = ForwardSearch()
            # task = self.brain.wm.task
            pos_init = frozenset(self.brain.motor.perceive())
            task = self.brain.wm.task.copy_with_new_initial_state(pos_init)
            full_init = task.initial_state
            if full_init in self.brain.ltm.unplannable_states:
                log.info("We know this state is unplannable. Skip right to exploration mode.")
                self.brain.affect.planning_impasse = True
                return
            subsumes = list(filter(lambda fs: state_subsumes(full_init, fs), self.brain.ltm.cached_plans.keys()))
            if len(subsumes) > 0:
                log.info("We know this state is plannable! Don't bother planning -- we already know this.")
                self.plan_exists = True
                self.brain.wm.current_plan = self.brain.ltm.cached_plans[subsumes[0]]
                return
            log.info("Goal: {}".format(task.goals))
            log.info("Searching for plan...")
            # frontier: the end of the road in this particular search
            # visited: all states visited
            plan, frontier, visited = forward_search.search(task)
            self.brain.wm.forward_states = visited
            log.info("Plan: {}".format(plan))
            # common, fluents = self._get_common_fluents(visited)
            self.brain.wm.clear_operator_stack()
            if plan:
                self.plan_exists = True
                self.brain.wm.current_plan = plan
            else:
                self.brain.affect.planning_impasse = True

    def _next(self):
        if self.plan_exists:
            return ExecutionMode(brain=self.brain)
        else:
            return ExplorationMode(brain=self.brain)

    def _get_common_fluents(self, visited, common_factor=3):
        """
        Returns a set of fluents present in at least "common_factor" states in visited
        """
        fluents = {}
        for state in visited:
            for fluent in state:
                try:
                    fluents[fluent] += 1
                except:
                    fluents[fluent] = 1

        common = []
        for fluent, count in fluents.items():
            if count >= common_factor:
                common.append(fluent)
        return common, fluents

