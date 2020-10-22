import gym
import gym_cupsworld

env = gym.make("CupsWorld-v0")
observation = env.reset()
for _ in range(1000):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)

  if done:
    print("resetting")
    observation = env.reset()
env.close()