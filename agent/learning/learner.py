from abc import abstractmethod, ABC
from collections import defaultdict
from functools import partial
from random import choice

import numpy as np

from agent.learning.policy import Policy


class Learner(ABC):
    @abstractmethod
    def __init__(self, env):
        self.env = env

    @abstractmethod
    def _train(self, s, a, r, sp): # Train this learner based on a transition or array of transitions.
        pass

    def train(self, s, a, r, sp):
        self._train(s, a, r, sp)


class TabularLearner(Learner, Policy):

    def _train_hashed(self, s, a, r, sp):
        self.Q[s][a] += self.learning_rate * (r + self.discount * np.max(self.Q[sp]) - self.Q[s][a])

    def _train(self, s, a, r, sp):
        hashable_s = self._hash_function(s)
        hashable_sp = self._hash_function(sp)
        self._train_hashed(hashable_s, a, r, hashable_sp)

    def _action(self, obs):
        q_entry = self.Q[self._hash_function(obs)]
        q_max = np.max(q_entry)
        return choice([a for a in range(len(q_entry)) if q_entry[a] == q_max])

    def __init__(self, env, hash_function, learning_rate=0.5, discount=0.99):
        self.env = env
        self._hash_function = hash_function
        self.learning_rate = learning_rate
        self.discount = discount
        self.Q = defaultdict(partial(np.zeros, self.env.action_space.n))


class CustomRewardTabularLearner(TabularLearner):
    def __init__(self, env, hash_function, rewarder, learning_rate=0.5, discount=0.99):
        super().__init__(env, hash_function, learning_rate, discount)
        self.reward_fn = rewarder

    def _train(self, s, a, r, sp):
        super()._train(s, a, self.reward_fn.reward(s,a,r,sp) if self.reward_fn is not None else r, sp)


def train_mult_steps(learner, s, a, r, sp, steps):
    base_discount = learner.discount
    learner.discount = base_discount ** max(steps, 1)
    learner.train(s, a, r, sp)
    learner.discount = base_discount