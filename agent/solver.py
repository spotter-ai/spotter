import csv
import logging
import pickle

import agent.agent_globals as params
from agent.brain import Brain

# Create and configure logger
from agent.planning_mode import PlanningMode

from helper import get_root_dir
logging.basicConfig(filename=get_root_dir() + "/logs/console.log",
                    format="%(name)s — %(levelname)s — %(message)s",
                    filemode='w')
log = logging.getLogger("agent")
log.setLevel(logging.DEBUG)


class Solver:

    def __init__(self, env, domain_bias, goal, detector, executor_class, brain=None, initial_episode_count=0,
                 executor=None, state_hasher=None, operators=None,
                 min_episodes=10, results_filename=None, pickle_filename=None,
                 operator_filename=None,
                 render=None,
                 checkpoint_every=500, log_every=100,
                 freeze_task=False, ops_every=1):
        if brain is None:
            self.brain = Brain(env=env, domain_bias=domain_bias, goal=goal, detector=detector,
                               executor_class=executor_class, state_hasher=state_hasher, executor=executor, render=render,
                               operators=operators, operator_filename=operator_filename, freeze_task=freeze_task,
                               ops_every=ops_every)
        else:
            self.brain = brain
        self.initial_mode = PlanningMode(brain=self.brain)
        self.mode = self.initial_mode
        self.episode_counter = initial_episode_count
        self.min_episodes = min_episodes
        self.results_filename = results_filename
        self.pickle_filename = pickle_filename
        self.checkpoint_every = checkpoint_every
        self.operator_filename = operator_filename
        self.log_every = log_every
        log.info("Agent is ready.")

    def solve(self):
        log.info("Solving...")
        self.brain.performance.needed_exploration = True
        while self.brain.episode_counter < self.min_episodes:
            # print("Episode: {}".format(self.episode_counter))
            log.info("Episode: {}".format(self.episode_counter))
            self.brain.performance.needed_exploration = False
            self.brain.performance.goal_achieved = False
            self.brain.motor.env.reset()
            self.brain.performance.cumulative_reward = 0
            self.mode = self.initial_mode
            self.brain.episode_counter = self.episode_counter
            while self.mode and not (self.brain.affect.giveup or self.brain.performance.goal_achieved):
                self.mode.run()
                self.mode = self.mode.next()
            # print("Reward: {}".format(self.brain.performance.cumulative_reward))
            # print("Step count: {}".format(self.brain.motor.env.step_count))
            if self.episode_counter % self.log_every == 0:
                print("{}/Episode {}: reward={}; # operators={}".format(
                                                                     self.brain.motor.env.spec.id,
                                                                     self.episode_counter,
                                                                     self.brain.performance.cumulative_reward,
                                                                     self.brain.performance.operators_discovered,
                                                                     ))
            if self.episode_counter % self.checkpoint_every == 0:
                self.output_pickle()
            self.brain.motor.increment_episode()
            self.episode_counter += 1
            data = [self.episode_counter, self.brain.performance.cumulative_reward, self.brain.motor.env.step_count,
                    True]
            self.output_results(data)

    def output_pickle(self):
        with open(self.pickle_filename, "wb") as file:
            pickle.dump((self.episode_counter, self.brain), file)

    def output_results(self, data):
        db_file_name = self.results_filename
        with open(db_file_name, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(data)

    def evaluate(self):
        log.info("Results")
        log.info("Goal Achieved: {}".format(self.brain.performance.goal_achieved))
        log.info("Number of Steps: {}".format(self.brain.performance.no_of_steps))

