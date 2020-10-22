import gym


class LastObsWrapper(gym.core.Wrapper):
    """
    Wrapper that allows you to see an Env's last observation.
    """
    def __init__(self, env):
        super().__init__(env)
        self.current_observation = None
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.current_observation = obs
        return obs
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.current_observation = obs
        return obs, reward, done, info
    def last_observation(self):
        return self.current_observation



