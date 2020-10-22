import gym
import logging

from agent.solver import Solver
from agent.execution.cupsworld_executor import CupsWorldExecutor
from agent.detection.cupsworld_detector import CupsWorldDetector

from helper import get_root_dir
logging.basicConfig(filename=get_root_dir() + "/logs/console.log",
                    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
                    filemode='w')

# Creating an object
log = logging.getLogger("Experiment")

# Setting the threshold of logger to DEBUG
log.setLevel(logging.DEBUG)


class Experiment:
    def __init__(self):
        pass

    def run(self):
        env = gym.make('CupsWorld-v0')
        domain = "domains/domain.pddl"
        domain_incorrect = "domains/domain_incorrect.pddl"
        goals = []
        goals.append("(on block_blue block_red)")
        goals.append("(on block_red cup)")
        goals.append("(and (on block_blue cup) (on block_red block_blue))") #spectacular fail because of path of gripper
        goals.append("(on cup block_red)")
        goals.append("(and (upright cup) (on block_blue block_red))")
        goals.append("(and (on cup block_red))")
        goals.append("(and (upright cup))")
        goals.append("(and (on cup block_blue) (upright cup))")
        goals.append("(and (holding cup) (inside block_red cup))")

        for goal in goals:
            print("Goal: ", goal)
            agent = Solver(env, domain_bias=domain_incorrect, goal=goal, detector=CupsWorldDetector, executor=CupsWorldExecutor)
            agent.solve()
            agent.evaluate()
            env.reset()


if __name__ == '__main__':
    expt = Experiment()
    expt.run()
    log.info("Done")
