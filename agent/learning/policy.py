from abc import ABC, abstractmethod
from random import randint, random


class Policy(ABC):

    def action(self, obs):
        return self._action(obs)

    @abstractmethod
    def _action(self, obs):
        pass


class EpsilonGreedyPolicy(Policy):

    def __init__(self, env, base_policy, epsilon):
        self.env = env
        self.num_actions = self.env.action_space.n
        self.base_policy = base_policy
        self.epsilon = epsilon

    def _action(self, obs):
        return self.base_policy.action(obs) if random() > self.epsilon else randint(0, self.num_actions-1)


class RandomDiscretePolicy(Policy):

    def __init__(self, env):
        self.env = env
        self.num_actions = self.env.action_space.n

    def _action(self, obs):
        return randint(0, self.num_actions-1)

