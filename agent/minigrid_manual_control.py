#!/usr/bin/env python3

import argparse
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window
import sys
sys.path.append('../')
from minigrid_envs.blocked_two_room import *
from agent.env_wrappers import NamedObjectWrapper, LastObsWrapper
from agent.detection.minigrid_detector import MiniGridDetector
from agent.execution.minigrid_executor import MiniGridExecutor


def redraw(img):
    if not args.agent_view:
        img = env.render('rgb_array', tile_size=args.tile_size)

    window.show_img(img)


def reset():
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)

    return obs


def step(action):
    obs, reward, done, info = executor.act(action)
    print('step=%s, reward=%.2f' % (env.step_count, reward))
    if done:
        print('done!')
        reset()
    else:
        redraw(obs)


def key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        step("(core_left)")
        return
    if event.key == 'right':
        step("(core_right)")
        return
    if event.key == 'up':
        step("(core_forward)")
        return

    # Spacebar
    if event.key == ' ':
        step("(core_toggle)")
        return
    if event.key == 'pageup':
        step("(core_pickup)")
        return
    if event.key == 'pagedown':
        step("(core_drop)")
        return

    if event.key == 'enter':
        step("(core_done)")
        return

    if event.key == 'k':
        step("(gotoobj1 agent " + key_id + ")")

    if event.key == 'd':
        step("(gotoobj1 agent " + door_id + ")")

    if event.key == 'u':
        step("(usekey agent " + key_id + ")")

    if event.key == "p": # pickup the key in particular
        step("(pickup agent " + key_id + ")")

    if event.key == "x":
        step("(putdown agent " + key_id + ")")


def get_key_id(obs):
    for object in obs['objects']:
        if obs['objects'][object]['encoding'][0] == 5:
            return object


def get_door_id(obs):
    for object in obs['objects']:
        if obs['objects'][object]['encoding'][0] == 4:
            return object


parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    default='MiniGrid-MultiRoom-N6-v0'
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=-1
)
parser.add_argument(
    "--tile_size",
    type=int,
    help="size at which to render tiles",
    default=32
)
parser.add_argument(
    '--agent_view',
    default=False,
    help="draw the agent sees (partially observable view)",
    action='store_true'
)

args = parser.parse_args()

# env = gym.make(args.env)
env = gym.make("MiniGrid-SpotterLevel3-v0")

# env = ReseedWrapper(env)

env = FullyObsWrapper(env)

env = NamedObjectWrapper(env)

env = LastObsWrapper(env)

detector = MiniGridDetector(env, domain="domains/minigrid.pddl")

executor = MiniGridExecutor(env, detector)

# if args.agent_view:
#     env = RGBImgPartialObsWrapper(env)
#     env = ImgObsWrapper(env)

window = Window('gym_minigrid - ' + args.env)
window.reg_key_handler(key_handler)

obs = reset()

key_id = get_key_id(obs)

door_id = get_door_id(obs)

print(key_id)
print(door_id)

# Blocking event loop
window.show(block=True)
