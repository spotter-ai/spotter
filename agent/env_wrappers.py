import gym
import copy

from gym import RewardWrapper
from gym.spaces import Discrete



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


class NamedObjectWrapper(gym.core.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_objs = 0

    def reset(self, **kwargs):
        self.num_objs = 0
        obs = self.env.reset(**kwargs)
        for i in range(self.grid.width):
            for j in range(self.grid.height):
                obj = self.grid.get(i, j)
                if obj is not None:
                    obj_type = obj.encode()[0]
                    if obj_type == 4:
                        obj.name = "door"
                    elif obj_type == 5:
                        obj.name = "key"
                    elif obj_type == 6:
                        obj.name = "ball"
                    elif obj_type == 8:
                        obj.name = "goal"
                    else:
                        obj.name = "obj" + str(self.num_objs).zfill(4)
                        self.num_objs += 1
        obs['objects'] = self._get_objs_dict(obs)
        return obs

    def _get_objs_dict(self, obs):
        objs_dict = {}
        for i in range(self.grid.width):
            for j in range(self.grid.height):
                obj = self.grid.get(i, j)
                if obj is not None:
                    objs_dict[obj.name] = {'x': i, 'y': j, 'encoding': obj.encode()}
        objs_dict['agent'] = {'x': self.agent_pos[0], 'y': self.agent_pos[1], 'encoding': (10, 0, self.agent_dir)}
        if self.carrying is not None:
            objs_dict[self.carrying.name] = {'x': -1, 'y': -1, 'encoding': self.carrying.encode()}
        return objs_dict

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs['objects'] = self._get_objs_dict(obs)
        return obs, reward, done, info


class ExecutorWrapper(gym.core.Wrapper):
    def __init__(self, env, domain_file, detector_class, executor_class, render=False, actions=None):
        super().__init__(env)
        self._detector = detector_class(env, domain_file)
        self._executor = executor_class(env, self._detector, render_mode="HUMAN" if render else None)
        if actions is None:
            self._actions = self._executor.all_actions()
        else:
            self._actions = actions
        self.action_space = Discrete(len(self._actions))
        self.trajectory = None

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self._executor.act(self._actions[action])
        if self._executor.accumulator:
            self.trajectory = self._executor.accumulator.trajectory()
        return obs, reward, done, info


class UnitRewardWrapper(RewardWrapper):
    def reward(self, reward):
        if reward > 0:
            return 1.
        return 0.


class RewardsWrapper(gym.core.Wrapper):
    '''
    This wrapper reimplements the following reward function
    +0.10 : Picking key
    +0.10 : Opening door
    +0.45 : dropping key
    +1.00 : Completing the task
    All the tasks are discounted wrt time steps
    '''
    def __init__(self, env):
        super().__init__(env)
        self.door_pos_x, self.door_pos_y = None, None
        self.drop_reward_count = 0

    def step(self, action):
        old_carry = copy.deepcopy(self.carrying)
        obs, reward, done, info = super().step(action)
        # Get the position in front of the agent
        fwd_pos = self.front_pos
        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)
        # compute discount
        discount = 0.9 * (self.step_count / self.max_steps)
        # HACKY method to get the door position
        for j in range(0, self.grid.height):
            for i in range(0, self.grid.width):
                cell = self.grid.get(i, j)
                if (cell.__class__.__name__ == 'Door'):
                    self.door_pos_x, self.door_pos_y = i, j
        # Picking key
        if action == self.actions.pickup:
            if (old_carry.__class__.__name__ != self.carrying.__class__.__name__
                    and self.carrying and self.carrying != self.obj
                    and not self.grid.get(self.door_pos_x, self.door_pos_y).is_open):
                reward = 0.10 - discount
        # Unlocking door
        elif action == self.actions.toggle:
            # HACK: Agent can only carry two things
            # key or final_object
            if (fwd_cell and fwd_cell != self.obj and \
                 self.carrying and self.carrying != self.obj and self.grid.get(self.door_pos_x, self.door_pos_y).is_open):
                reward = 0.10 - discount
        # Drop key after door unlock
        elif action == self.actions.drop:
            # door is open, agent has dropped key
            # and give drop reward only once
            if self.grid.get(self.door_pos_x, self.door_pos_y).is_open \
                    and self.carrying is None and self.drop_reward_count < 1:
                reward = 0.45 - discount
                self.drop_reward_count +=1
        return obs, reward, done, info

    def reset(self):
        obs = super().reset()
        self.drop_reward_count = 0
        return obs


if __name__ == "__main__":
    import gym

    from gym_minigrid.wrappers import FullyObsWrapper, ReseedWrapper

    from agent.execution.minigrid_executor import MiniGridExecutor
    from agent.detection.minigrid_detector import MiniGridDetector

    env = gym.make('MiniGrid-UnlockPickup-v0')
    env = ReseedWrapper(env)
    env = FullyObsWrapper(env)
    env = NamedObjectWrapper(env)
    env = LastObsWrapper(env)
    env = ExecutorWrapper(env, "../domains/gridworld_abstract.pddl", MiniGridDetector, MiniGridExecutor)
    env = RewardsWrapper(env)
    print(env.action_space.n)

    for i in range(env.action_space.n):
        env.step(i)
        env.reset()

