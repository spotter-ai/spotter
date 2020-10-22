import time

import gym
import gym_cupsworld
from gym.wrappers import FlattenObservation

from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.dqn import MlpPolicy


def train(env, type, timesteps):
    env.reset()
    print(check_env(env))
    env = FlattenObservation(env)
    print(env.reward_range)
    print(env.action_space)
    if type == "DQN":
        model = DQN('MlpPolicy', exploration_fraction=0.999, env=env, verbose=1)
    elif type == "A2C":
        model = A2C('MlpPolicy', env=env, verbose=1)
    elif type == "PPO":
        model = PPO('MlpPolicy', env=env, verbose=1)

    model.learn(total_timesteps=timesteps)
    model.save("model_cups")


def act(env, model):
    # env is deterministic as in if I say "go right" the gripper will go right all the time.
    obs = env.reset()
    for i in range(100):
        env.render()
        action, _states = model.predict(obs, deterministic=True)
        # print(action)
        obs, reward, done, info = env.step(action)
        if done:
            print('[FINAL] obs=', obs, 'reward=', reward, 'done=', done)
            break

type = "DQN"
TIME_STEPS = 50000
env = gym.make('CupsWorld-v0')
# train(env, type, TIME_STEPS)
if type == "A2C":
    model = A2C.load('model_cups')
elif type == "DQN":
    model = DQN.load('model_cups')
elif type == "PPO":
    model = PPO.load('model_cups')
act(env, model)
env.close()