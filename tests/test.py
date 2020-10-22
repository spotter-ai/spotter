import gym
import gym_cupsworld
import time

from gym.wrappers import FlattenObservation
from gym_cupsworld.envs.wrappers import LastObsWrapper
from stable_baselines3.common.env_checker import check_env



def random(env, number):
    plan = []
    for i in range(number):
         if i % 5 == 0:
             plan.append(env.action_space.sample())
         else:
             plan.append(env.actions.nop)
    return plan


def flip_cup(env):
    plan = []
    #Move right to cup
    for _ in range(35):
        plan.append(env.actions.right)
    for _ in range(9):
        plan.append(env.actions.down)
    plan.append(env.actions.grasp)
    for _ in range(40):
        plan.append(env.actions.up)
    for i in range(10):
        plan.append(env.actions.nop)
    for _ in range(19):
        plan.append(env.actions.left)
    for i in range(10):
        plan.append(env.actions.nop)
    plan.append(env.actions.release)
    for i in range(50):
        plan.append(env.actions.nop)
    # for _ in range(15):
    #     plan.append(env.actions.down)
    # plan.append(env.actions.grasp)
    # for _ in range(30):
    #     plan.append(env.actions.right)
    for _ in range(55):
        plan.append(env.actions.nop)
    return plan


def move_with_cup(env):
    plan = []
    #Move right to cup
    for _ in range(35):
        plan.append(env.actions.right)
    for _ in range(9):
        plan.append(env.actions.down)
    plan.append(env.actions.grasp)
    for _ in range(30):
        plan.append(env.actions.up)
    # for i in range(10):
    #     plan.append(env.actions.nop)
    for _ in range(30):
        plan.append(env.actions.left)
    # for i in range(10):
    #     plan.append(env.actions.nop)
    plan.append(env.actions.release)
    for i in range(30):
        plan.append(env.actions.nop)
    for _ in range(29):
        plan.append(env.actions.down)
    plan.append(env.actions.grasp)
    for _ in range(30):
        plan.append(env.actions.right)
    for _ in range(15):
        plan.append(env.actions.nop)
    return plan


def pick_and_stack_plan(env):
    plan = []
    init = [env.actions.down, env.actions.down, env.actions.grasp]
    plan.extend(init)
    for _ in range(3):
        plan.append(env.actions.right)
    for i in range(12):
        plan.append(env.actions.down)
    k = plan.append(env.actions.grasp)
    for i in range(30):
        plan.append(env.actions.up)
    for i in range(30):
        plan.append(env.actions.nop)
    for i in range(20):
        plan.append(env.actions.left)
    for i in range(10):
        plan.append(env.actions.nop)
    plan.append(env.actions.release)
    for i in range(30):
        plan.append(env.actions.nop)
    return plan


def execute(env, plan):
    global HUMAN
    for a in plan:
        if HUMAN:
            env.render()
        else:
            env.render('console')
        obs, reward, done, info = env.step(a)
        if done:
            print('[FINAL] obs=', obs, 'reward=', reward, 'done=', done)
            break
        print('obs=', obs, 'reward=', reward, 'done=', done)
        # time.sleep(0.3)


def execute_reset(env, plan):
    global HUMAN
    for a in plan:
        if HUMAN:
            env.render()
        else:
            env.render('console')
        obs, reward, done, info = env.step(a)
        if done:
            print('[FINAL] obs=', obs, 'reward=', reward, 'done=', done)
            env.reset()
        print('obs=', obs, 'reward=', reward, 'done=', done)


def main():
    env = gym.make('CupsWorld-v0')
    env = LastObsWrapper(env)
    obs = env.reset()
    # plan = pick_and_stack_plan(env)
    # plan = random(env,1000)
    # plan = move_with_cup(env)
    plan = flip_cup(env)
    execute(env, plan)
    # execute_reset(env, plan)
    print("====")
    print(env.last_observation())
    env.close()


HUMAN = True
main()
