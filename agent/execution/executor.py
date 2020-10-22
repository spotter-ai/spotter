from abc import abstractmethod, ABC


class Executor(ABC):
    def __init__(self):
        pass

    def all_actions(self):
        return self._all_actions()

    @abstractmethod
    def _all_actions(self):
        pass

    @abstractmethod
    def _act(self, action):
        pass

    @abstractmethod
    def _remove_learner(self, learner):
        pass

    def remove_learner(self, learner):
        self._remove_learner(learner)

    @abstractmethod
    def _attach_learner(self, learner):
        pass

    def attach_learner(self, learner):
        self._attach_learner(learner)

    def clear_learners(self):
        self._clear_learners()

    @abstractmethod
    def _clear_learners(self):
        pass

    def act(self, action):
        return self._act(action)

    @abstractmethod
    def _execute_core_action(self, action, accumulator):
        pass

    def execute_core_action(self, action, accumulator=None):
        return self._execute_core_action(action, accumulator)


def operator_parts(operator_name):
    """
    input: "(pickup cup)"
    output: "pickup", ["cup"]
    """
    rem_parens = operator_name.replace("(","").replace(")","")
    return rem_parens.split(" ")[0], rem_parens.split(" ")[1:]


class RewardAccumulator:

    def accumulate(self, action, obs, reward, done, info):
        self._trajectory.append((self.last_obs, action, reward, obs))
        if self.discount == 1.:
            self.accum_reward += reward
        else:
            self.accum_reward += (self.discount ** self.num_actions) * reward
        self.num_actions += 1
        self.last_obs = obs
        self.last_info = info
        self.done = done
        if self.num_actions >= self.action_timeout or done:
            raise EpisodeTerminatedException
        return obs, self.accum_reward, done, info

    def combine_steps(self):
        return self.last_obs, self.accum_reward, self.done, self.last_info

    def trajectory(self):
        return self._trajectory

    def __init__(self, env, action_timeout=10000, discount=1.):
        self.env = env
        self.accum_reward = 0.
        self.discount = discount
        self.done = False
        self.num_actions = 0
        self.last_obs = env.last_observation()
        self.last_info = None
        self.action_timeout = action_timeout
        self._trajectory = []


class EpisodeTerminatedException(Exception):
    pass
