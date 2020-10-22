import gym
import pddlgym
import imageio

env = gym.make("PDDLEnvBlocks-v0")
obs, debug_info = env.reset()
print(obs)
for _ in range(3):
    env.render()
    action = env.action_space.sample(obs)
    obs, reward, done, debug_info = env.step(action)
    print(obs)
env.close()
