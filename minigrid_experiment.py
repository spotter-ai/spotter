import math
import os
from random import seed

import pickle
import uuid

import argparse

import gym
import logging
import numpy as np

from gym_minigrid.wrappers import FullyObsWrapper

from agent.learning.learner import TabularLearner, CustomRewardTabularLearner, train_mult_steps
from agent.learning.policy import EpsilonGreedyPolicy
from helper import get_root_dir
import minigrid_envs

from agent.detection.minigrid_detector import MiniGridDetector
from agent.execution.minigrid_executor import MiniGridExecutor
from agent.solver import Solver

from agent.env_wrappers import NamedObjectWrapper, LastObsWrapper, RewardsWrapper, UnitRewardWrapper
from agent.statehashing.minigrid_hashing import MinigridStateHasher

ROOT_DIR = get_root_dir()

logging.basicConfig(filename=ROOT_DIR + "/logs/console.log",
                    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
                    filemode='w')

# Creating an object
log = logging.getLogger("Experiment")

# Setting the threshold of logger to DEBUG
log.setLevel(logging.DEBUG)

NUM_TO_NAME_AND_GOAL = {'1': ("MiniGrid-SpotterLevel1-v0", "(open door)"),
                        '2': ("MiniGrid-SpotterLevel2-v0", "(open door)"),
                        '3': ("MiniGrid-SpotterLevel3-v0", "(atgoal agent goal)"),
                        'unblocked': ("MiniGrid-UnblockedGoal-v0", "(atgoal agent goal)"),
                        'test': ("MiniGrid-SpotterTestBlocked-v0", "(open door)")}

BASE_ACTIONS = ["(core_left)", "(core_right)", "(core_forward)", "(core_pickup)", "(core_drop)", "(core_toggle)"]

RELEVANT_ACTIONS = {'1': BASE_ACTIONS + ["(gotoobj1 agent key)", "(gotodoor1 agent door)"],
                    '2': BASE_ACTIONS + ["(gotoobj1 agent key)", "(gotodoor1 agent door)"],
                    '3': BASE_ACTIONS + ["(gotoobj1 agent key)", "(gotodoor1 agent door)",
                                         "(enterroomof agent door goal)",
                                         "(gotoobj2 agent goal)",
                                         "(stepinto agent goal)"]}

ALPHA = 0.1
GAMMA = 0.99
MAX_EPSILON = 0.90
MIN_EPSILON = 0.05
EXPLORATION_STOP = 1000000
LAMBDA = -math.log(0.01) / EXPLORATION_STOP # speed of decay

def get_operator_filename(level, run_id):
    return ROOT_DIR + "/results/spotter/" + str(level) + "/operators/" + str(level) + "_" + str(run_id) + "_ops.txt"


def get_results_filename(level, run_id):
    return ROOT_DIR + "/results/spotter/" + str(level) + "/" + str(level) + "_" + str(run_id) + ".csv"


def get_pickle_file(level, run_id):
    return ROOT_DIR + "/results/spotter/" + str(level) + "/pickles/brain_" + str(level) + "_" + run_id + ".pickle"


def run_transfer(orig_level, transfer_level, transfer_goal, run_id, domain_file, min_episodes, render,
                 checkpoint_every, log_every, ops_every):
    orig_pickle_filename = get_pickle_file(orig_level, run_id)
    agent_render = 'HUMAN' if render else None
    with open(orig_pickle_filename, "rb") as file:
        eps_so_far, brain = pickle.load(file)
        agent = Solver(None, None, None, None, None, brain, eps_so_far, None, MinigridStateHasher, None,
                       0,
                       "",
                       "", get_operator_filename(orig_level, run_id),
                       render=None,
                       ops_every=ops_every
                       )
        inherited_executor = agent.brain.motor.executor
        inherited_executor.clear_learners()
        inherited_operators = [op for op in agent.brain.wm.task.operators if op.name.startswith("new_action")]
        num_new_ops = 0
        for op in inherited_operators:
            inherited_executor.rename_op(op, "transfer_" + str(num_new_ops).zfill(4))
            num_new_ops += 1
        env = gym.make(transfer_level)
        env.seed(seed=seed())
        env = FullyObsWrapper(env)
        env = NamedObjectWrapper(env)
        env = LastObsWrapper(env)
        env.reset()
        os.makedirs("results" + os.sep + "spotter" + os.sep + str(transfer_level) + os.sep + "operators",
                    exist_ok=True)
        os.makedirs("results" + os.sep + "spotter" + os.sep + str(transfer_level) + os.sep + "pickles",
                    exist_ok=True)
        inherited_executor.set_environment(env)
        results_filename = get_results_filename(transfer_level, run_id)
        pickle_filename = get_pickle_file(transfer_level, run_id)
        agent_render = 'HUMAN' if render else None
        agent = Solver(env, domain_bias=domain_file, goal=transfer_goal, detector=MiniGridDetector,
                       executor_class=MiniGridExecutor, state_hasher=MinigridStateHasher,
                       executor=inherited_executor, operators=inherited_operators,
                       min_episodes=min_episodes, results_filename=results_filename,
                       operator_filename=get_operator_filename(transfer_level, run_id),
                       pickle_filename=pickle_filename, render=agent_render, checkpoint_every=checkpoint_every,
                       log_every=log_every)
        agent.solve()
        agent.evaluate()
        # Final pickle of the agent's brain at the conclusion of the episode
        with open(pickle_filename, "wb") as file:
            pickle.dump((agent.episode_counter, agent.brain), file)


def resume_experiment(level, run_id, new_min_episodes, render, checkpoint_every, log_every):
    pickle_filename = get_pickle_file(level, run_id)
    agent_render = 'HUMAN' if render else None
    with open(pickle_filename, "rb") as file:
        eps_so_far, brain = pickle.load(file)
        agent = Solver(None, None, None, None, None, brain, eps_so_far, None, MinigridStateHasher, None,
                       new_min_episodes,
                       get_results_filename(level, run_id),
                       pickle_filename, get_operator_filename(level, run_id),
                       render=agent_render, checkpoint_every=checkpoint_every, log_every=log_every
                       )
    if agent:
        agent.solve()
        with open(get_pickle_file(level, run_id), "wb") as file:
            pickle.dump((agent.episode_counter, agent.brain), file)


def run_baseline_episode(env, learner, episode, train_lower):
    epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * \
              math.exp(-LAMBDA * episode)
    done = False
    s = env.reset()
    cum_reward = 0.
    policy = EpsilonGreedyPolicy(env, learner, epsilon)
    while not done:
        a = policy.action(s)
        last_step_count = env.step_count
        sp, reward, done, info = env.step(a)
        cum_reward += reward
        num_steps = env.step_count - last_step_count
        train_mult_steps(learner, s, a, reward, sp, num_steps)
        if train_lower and num_steps > 1:
            for s1, a1, r1, sp1 in env.trajectory:
                learner.train(s1, a1, r1, sp1)
        s = sp
    return cum_reward, epsilon


def baseline_results_file(level, run_id, augmented, train_lower):
    return ROOT_DIR + "/results/baseline/" + str(level) + "/" + level + "_" + str(run_id) + \
           ("_aug" if augmented else "") + ("_lower" if train_lower else "") + ".csv"


def baseline_pickle_file(level, run_id, augmented, train_lower):
    return ROOT_DIR + "/results/baseline/" + str(level) + "/pickles/Q_" + str(level) + "_" + run_id + \
           ("_aug" if augmented else "") + ("_lower" if train_lower else "") + ".pickle"


def save_to_file(filename, data):
    import csv
    with open(filename, "a") as file:
        writer = csv.writer(file)
        writer.writerow(data)


class UnitRewarder:
    def reward(self, s, a, r, sp):
        return 1. if r > 0. else 0.


def resize_q(q, new_action_size):
    return {k: np.concatenate((v, np.zeros(new_action_size - len(v))), axis=0) for (k, v) in q.items()}


def run_consecutive_baseline_levels(level_info, domain_file, augment_actions,
                                    render=False, checkpoint_every=500, log_every=100, train_lower=False):
    run_id = uuid.uuid4().hex
    q = {}
    for level, min_episodes, actions in level_info:
        os.makedirs(ROOT_DIR + "/results/baseline/" + str(level) + "/pickles/", exist_ok=True)
        env = get_minigrid_environment(level, domain_file=domain_file, use_executor=augment_actions, render=render,
                                       actions=actions)
        hasher = MinigridStateHasher(MiniGridDetector(env,domain_file))
        results_file = baseline_results_file(level, run_id, augment_actions, train_lower)
        pickle_filename = baseline_pickle_file(level, run_id, augment_actions, train_lower)
        ql = CustomRewardTabularLearner(env, hasher.hash, rewarder=UnitRewarder(), learning_rate=0.1)

        ql.Q.update(resize_q(q, env.action_space.n))
        for episode_num in range(min_episodes):
            cum_reward, epsilon = run_baseline_episode(env, ql, episode_num, train_lower)
            save_to_file(results_file, [episode_num, env.step_count, cum_reward, epsilon])
            if episode_num % log_every == 0:
                print("{}/Episode {}: reward={}".format(level, episode_num, cum_reward))
            if episode_num % checkpoint_every == 0:
                with open(pickle_filename, "wb") as file:
                    pickle.dump((episode_num, ql), file)
        q = dict(ql.Q)


def run_consecutive_levels(level_goal_episodes, domain_file, render=False, checkpoint_every=500, log_every=100,
                           freeze_task=False, ops_every=1):
    inherited_executor = None
    inherited_operators = None
    num_new_ops = 0
    run_id = uuid.uuid4().hex
    for level, goal, min_episodes in level_goal_episodes:
        env = gym.make(level)
        env.seed(seed=seed())
        env = FullyObsWrapper(env)
        env = NamedObjectWrapper(env)
        env = LastObsWrapper(env)
        env.reset()
        os.makedirs("results" + os.sep + "spotter" + os.sep + str(level) + os.sep + "operators", exist_ok=True)
        os.makedirs("results" + os.sep + "spotter" + os.sep + str(level) + os.sep + "pickles", exist_ok=True)
        if inherited_executor:
            inherited_executor.set_environment(env)
        results_filename = get_results_filename(level, run_id)
        pickle_filename = get_pickle_file(level, run_id)
        agent_render = 'HUMAN' if render else None
        agent = Solver(env, domain_bias=domain_file, goal=goal, detector=MiniGridDetector,
                       executor_class=MiniGridExecutor, state_hasher=MinigridStateHasher,
                       executor=inherited_executor, operators=inherited_operators,
                       min_episodes=min_episodes, results_filename=results_filename,
                       operator_filename=get_operator_filename(level, run_id),
                       pickle_filename=pickle_filename, render=agent_render, checkpoint_every=checkpoint_every,
                       log_every=log_every, freeze_task=freeze_task, ops_every=ops_every)
        agent.solve()
        agent.evaluate()
        # Final pickle of the agent's brain at the conclusion of the episode
        with open(pickle_filename, "wb") as file:
            pickle.dump((agent.episode_counter, agent.brain), file)
        inherited_executor = agent.brain.motor.executor
        inherited_executor.clear_learners()
        inherited_operators = [op for op in agent.brain.wm.task.operators if op.name.startswith("new_action")]
        for op in inherited_operators:
            inherited_executor.rename_op(op, "transfer_" + str(num_new_ops).zfill(4))
            num_new_ops += 1


def get_door_id(env):
    obs = env.last_observation()
    for obj in obs['objects']:
        if obs['objects'][obj]['encoding'][0] == 4:
            return obj


def get_minigrid_environment(environment_name='MiniGrid-UnlockPickup-v0', domain_file="domains/gridworld_abstract.pddl",
                             render=False, use_executor=False, actions=None):
    from agent.env_wrappers import ExecutorWrapper
    env = gym.make(environment_name)
    env.seed(seed=seed())
    env = FullyObsWrapper(env)
    env = NamedObjectWrapper(env)
    env = LastObsWrapper(env)
    # TODO note: this is a relative path, so this code needs to be run from a file in the uppermost directory.
    # if you want a different relative path, you'll have to specify it yourself.
    if use_executor and actions is not None:
        env = ExecutorWrapper(env, domain_file, MiniGridDetector, MiniGridExecutor, render, actions)
    return env


parser = argparse.ArgumentParser()
parser.add_argument(
    '--level',
    nargs='*',
    default=['1', '2', '3']
)

parser.add_argument(
    '--transfer_level'
)

parser.add_argument(
    '--checkpoint_every',
    type=int,
    default=500
)

parser.add_argument(
    '--log_every',
    type=int,
    default=100
)

parser.add_argument(
    '--ops_every',
    type=int,
    default=1
)

parser.add_argument(
    '--baseline',
    default=False,
    help="Run Q-learning baselines.",
    action="store_true"
)

parser.add_argument(
    '--augmented',
    default=False,
    help="Run Q-learning baselines with the planner-augmented action space.",
    action="store_true"
)

parser.add_argument(
    '--train_lower',
    default=False,
    help="Set this to true if you want an augmented baseline to train on lower-level actions of executed HLAs.",
    action="store_true"
)

parser.add_argument(
    '--episodes',
    nargs='*',
    type=int,
    default=[1E4, 1E4, 1E4]
)

parser.add_argument(
    '--domain',
    default=get_root_dir() + "/domains/minigrid.pddl"
)

parser.add_argument(
    '--render',
    default=False,
    help="draw the agent sees (partially observable view)",
    action='store_true'
)

parser.add_argument(
    '--runs',
    default=1,
    type=int
)

parser.add_argument(
    '--resume',
    default=False,
    help="resume a level/run combination (useful for debugging)",
    action="store_true"
)

parser.add_argument(
    '--id',
    help="ID of run to resume (only useful if --resume is on)"
)

parser.add_argument(
    '--freeze_task',
    default=False,
    help="True if the agent shouldn't actually *add* operators it's learnerd",
    action="store_true"
)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.baseline:
        level_info = []
        for i in range(len(args.level)):
            level, goal = NUM_TO_NAME_AND_GOAL[args.level[i]]
            actions = RELEVANT_ACTIONS[args.level[i]] if args.augmented else None
            level_info.append((level, args.episodes[i], actions))
        for i in range(args.runs):
            log.info("Starting baseline run {}".format(i))
            run_consecutive_baseline_levels(level_info, args.domain, args.augmented, args.render, args.checkpoint_every,
                                            args.log_every, args.train_lower)
    else:
        if args.resume:
            for i in range(len(args.level)):
                level, goal = NUM_TO_NAME_AND_GOAL[args.level[i]]
                resume_experiment(level, args.id, args.episodes[i], args.render, args.checkpoint_every, args.log_every)
        elif args.transfer_level:
            transfer_level, transfer_goal = NUM_TO_NAME_AND_GOAL[args.transfer_level]
            orig_level, orig_goal = NUM_TO_NAME_AND_GOAL[args.level[0]]
            run_transfer(orig_level, transfer_level, transfer_goal, args.id, args.domain, args.episodes[0], args.render,
                             args.checkpoint_every, args.log_every, args.ops_every)
        else:
            level_goal_episodes = []
            for i in range(len(args.level)):
                level, goal = NUM_TO_NAME_AND_GOAL[args.level[i]]
                level_goal_episodes.append((level, goal, args.episodes[i]))
            for i in range(args.runs):
                log.info("Starting run {}".format(i))
                run_consecutive_levels(level_goal_episodes, args.domain, args.render, args.checkpoint_every, args.log_every,
                                           args.freeze_task, args.ops_every)
    log.info("Done")
