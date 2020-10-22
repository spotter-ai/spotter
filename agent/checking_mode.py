from agent.mode import Mode

import logging

from helper import get_root_dir
logging.basicConfig(filename=get_root_dir() + "/logs/console.log",
                    format="%(name)s — %(levelname)s — %(message)s",
                    filemode='w')
log = logging.getLogger("checking_mode")
log.setLevel(logging.DEBUG)


class CheckingMode(Mode):
    """
    Checks for blocking issues
    """

    def __init__(self, brain):
        super().__init__(brain)
        self.name = "Checking Mode"
        self.situation = self.brain.wm.situation
        self.replan = False
        self.reexecuted = False

    def _run(self):
        # Operator stack is empty and the agent has tried things

        # Preconditions of next operator not met in state
        next_op = self.brain.wm.see_next_operator()
        if next_op:
            if next_op.applicable(set(self.brain.motor.perceive())):
                log.info("We should be fine. Preconditions for next op satisfied now")
                self.brain.affect.blocked = False
            else:
                log.info("Execution Impasse: Preconditions for next operator not satisfied")
                log.info("Positive preconditions not met: {}".format(next_op.pos_preconditions
                                                                     - set(self.brain.motor.perceive())))
                log.info("Negative preconditions not met: {}".format(next_op.neg_preconditions
                                                                     & set(self.brain.motor.perceive())))

                # log.info("Trying to run operator again...")
                # if not self.reexecuted:
                #     pass
                # dkasenberg: We're not going to repeat operators so much as replan again.

                self.brain.affect.blocked = True
                self.brain.affect.execution_impasse = True
                self.brain.wm.clear_operator_stack()
        else:
            if len(self.brain.ltm.history) > 1:
                # check if goal is satisfied:
                if self.brain.wm.task.goal_reached(set(self.brain.motor.perceive())):
                    self.brain.performance.goal_achieved = True
                    log.info("Goal reached. Done")
                    self.brain.affect.blocked = False
                else:
                    log.info("Goal Impasse: No operators to try and I'm not a noob")
                    self.brain.affect.blocked = True
                    self.brain.affect.execution_impasse = True
                    self.brain.wm.clear_operator_stack()
            else:
                # empty history and stack suggests that the agent should replan
                pass

    def _next(self):
        if self.brain.affect.blocked:
            if self.replan:
                from agent.planning_mode import PlanningMode
                return PlanningMode(self.brain)
                # return ReplacingMode(self.brain)
            else:
                from agent.exploration_mode import ExplorationMode
                return ExplorationMode(self.brain)
                # pass
                # return ReplacingMode(self.brain)
        else:
            from agent.execution_mode import ExecutionMode
            return ExecutionMode(self.brain)





