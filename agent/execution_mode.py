from agent.mode import Mode

import logging

from helper import get_root_dir
logging.basicConfig(filename=get_root_dir() + "/logs/console.log",
                    format="%(name)s — %(levelname)s — %(message)s",
                    filemode='w')
log = logging.getLogger("execution_mode")
log.setLevel(logging.INFO)

import agent.agent_globals as globals


class ExecutionMode(Mode):

    def __init__(self, brain):
        self.name = "Execution Mode"
        self.terminated_episode = False
        super().__init__(brain)

    def _run(self):
        result = self.brain.wm.get_next_operator()
        if result:
            operator, idx = result
            log.info("Executing operator: {}".format(operator.name))
            next_fluent_state, reward, done, info = self.brain.motor.act(operator.name)
            self.brain.performance.cumulative_reward += reward
            if done:
                self.terminated_episode = True
                self.brain.wm.clear_operator_stack()
                log.info("The environment terminated the episode. How rude!")
            log.debug("Fluent State: {}".format(next_fluent_state))
            prev_fluent_state = self.brain.ltm.recall_last()[1]
            self.brain.wm.situation = {"prev": prev_fluent_state, "op": operator, "op_idx": idx, "next": next_fluent_state}
            log.debug("Situation: {}".format(self.brain.wm.situation))
            self.brain.ltm.add_to_history((operator, next_fluent_state))
            self.brain.performance.increment_step()

    def _next(self):
        # Transition to next state
        if self.terminated_episode:
            # self.brain.motor.env.reset()
            return None
            # from agent.planning_mode import PlanningMode
            # return PlanningMode(brain=self.brain)
        else:
            from agent.checking_mode import CheckingMode
            return CheckingMode(brain=self.brain)
