import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pygame
import pymunk
import pymunk.pygame_util
import numpy as np
from enum import IntEnum
import time


class Cup(pygame.sprite.Sprite):
    def __init__(self, pos, space, orientation, identifier, color):
        super().__init__()
        CUP_MASS = 0.1  # 0.2
        CUP_WIDTH = 35  # 70
        CUP_HEIGHT = 40  # 90
        CUP_FRICTION = 0.7
        WALL_THICKNESS = 2  # 3

        self.space = space
        self.name = 'cup'
        self.id = identifier

        # Corners of the cup (relative position - relative to center of cup)
        a = (-CUP_WIDTH / 2, -CUP_HEIGHT / 2)
        b = (-CUP_WIDTH / 2, CUP_HEIGHT / 2)
        c = (CUP_WIDTH / 2, CUP_HEIGHT / 2)
        d = (CUP_WIDTH / 2, -CUP_HEIGHT / 2)

        # Pymunk definitions
        self.body = pymunk.Body()
        self.body.position = pos
        self.initial_position = pos
        self.side1 = pymunk.Segment(None, a, b, WALL_THICKNESS)
        self.side1.body = self.body
        self.side1.mass = CUP_MASS
        self.side1.friction = CUP_FRICTION
        self.side1.color = color
        self.side1.collision_type = 1

        self.base = pymunk.Segment(None, b, c, WALL_THICKNESS)
        self.base.body = self.body
        self.base.mass = CUP_MASS
        self.base.friction = CUP_FRICTION
        self.base.color = color
        self.base.collision_type = 1

        self.side2 = pymunk.Segment(None, c, d, WALL_THICKNESS)
        self.side2.body = self.body
        self.side2.mass = CUP_MASS
        self.side2.friction = CUP_FRICTION
        self.side2.color = color
        self.side2.collision_type = 1

        self.body.angle = orientation
        self.space.add(self.body, self.base, self.side1, self.side2)
        self.space.reindex_shapes_for_body(self.body)

        self.orientation = self.body.angle * 180 / np.math.pi


class Platform(pygame.sprite.Sprite):
    def __init__(self, pos, space, identifier):
        super().__init__()
        START_POS = (0, 10)  # (50, 50)
        END_POS = (200, 10)  # (750, 50)
        THICKNESS = 5  # 20
        ELASTICITY = 0.5  # 0.3
        FRICTION = 0.6
        MASS = 5
        self.space = space
        self.name = "platform"
        self.id = identifier

        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = pos
        self.shape = pymunk.Segment(self.body, START_POS, END_POS, THICKNESS)
        self.shape.elasticity = ELASTICITY
        self.shape.friction = FRICTION
        self.shape.collision_type = 3  # 3 for table

        self.space.add(self.body, self.shape)


class Block(pygame.sprite.Sprite):
    def __init__(self, pos, space, color):
        super().__init__()
        self.name = 'block'
        BLOCK_MASS = 2  # 0.3
        BLOCK_DENSITY = 3
        BLOCK_SIZE = (27, 27)  # (50, 50)
        BLOCK_ELASTICITY = 0.5
        BLOCK_FRICTION = 0.5
        RADIUS = 0.8  # ???

        self.space = space

        # Pymunk aspects
        inertia = pymunk.moment_for_box(BLOCK_MASS, BLOCK_SIZE)
        self.body = pymunk.Body(BLOCK_MASS, inertia)
        self.body.position = pos
        self.initial_position = pos
        self.shape = pymunk.Poly.create_box(self.body, size=BLOCK_SIZE, radius=RADIUS)
        self.shape.elasticity = BLOCK_ELASTICITY
        self.shape.friction = BLOCK_FRICTION
        self.shape.color = color
        self.shape.collision_type = 2

        self.space.add(self.body, self.shape)


class Gripper(pygame.sprite.Sprite):
    def __init__(self, pos, space, identifier):
        super().__init__()
        GRIPPER_SIZE = (30, 5)  # (80, 20)
        SPEED = 400
        FRICTION = 0.6
        super().__init__()
        self.name = "gripper"
        self.id = identifier
        self.gripper_size = GRIPPER_SIZE
        self.speed = SPEED
        self.space = space
        self.body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.body.position = pos
        self.initial_position = pos
        self.shape = pymunk.Poly.create_box(self.body, size=self.gripper_size)
        self.shape.collision_type = 4  # collision type for gripper = 4
        self.shape.color = pygame.color.THECOLORS["black"]
        self.shape.friction = FRICTION
        self.grasped = 0  # 0: not grasped object, #1: grasped object

        self.space.add(self.body, self.shape)


class CupsWorldEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        nop = 0
        left = 1
        right = 2
        up = 3
        down = 4
        # Grasp
        grasp = 5
        # release
        release = 6

    def __init__(self):
        print("Initializing Environment")
        WIDTH = 200  # 800
        HEIGHT = 150  # 600
        self.size = (WIDTH, HEIGHT)
        GRIPPER_STEP_SIZE = 2  # 10
        DELAY = 0
        MAX_STEPS = 1000
        WAIT_FOR_PHYSICS = 20

        self.delay_bw_actions = DELAY
        self.gripper_step_size = GRIPPER_STEP_SIZE

        # Initialize pymunk space
        self.space = pymunk.Space()
        self.space.gravity = (0.0, -2000)
        self.space.sleep_time_threshold = 0.1  # not sure i need this.

        # Make Gripper (kinematic object)
        self.gripper = Gripper((WIDTH / 3, HEIGHT / 2), self.space, 4)  # this also adds it to the pymunk space
        # self.gripper_group = pygame.sprite.Group()
        # self.gripper_group.add(self.gripper)

        # Make Platform (static object)
        self.platform1 = Platform((0, 0), self.space, 3)

        # Make Block (dynamic object)
        self.block1 = Block((WIDTH / 8, HEIGHT / 4), self.space, pygame.color.THECOLORS['cornflowerblue'])
        self.block2 = Block((WIDTH / 3, HEIGHT / 4), self.space, pygame.color.THECOLORS['salmon'])

        # Make cup (dynamic objecT)
        self.cup = Cup((WIDTH / 1.5, HEIGHT / 4), self.space, 0, 4, pygame.color.THECOLORS['purple'])

        # Sprite groups
        self.sprite_group = pygame.sprite.Group()
        self.sprite_group.add(self.cup)
        self.sprite_group.add(self.block1)
        self.sprite_group.add(self.block2)

        # Initialize gym action space and observation space
        self.actions = CupsWorldEnv.Actions
        self.action_space = spaces.Discrete(len(self.actions))

        self.reward_range = (0,1)

        #  Observation space
        # gripper = spaces.Dict({
        #     'position': spaces.Box(np.array([0, 0]), high=np.array([WIDTH, HEIGHT]), shape=(2,), dtype=np.int8),
        #     'state': spaces.Discrete(2),
        # })
        #
        # cup = spaces.Dict({
        #     'position': spaces.Box(np.array([0, 0]), high=np.array([WIDTH, HEIGHT]), shape=(2,), dtype=np.int8),
        #     'state': spaces.Discrete(4),
        # })
        #
        # block_red = spaces.Dict({
        #     'position': spaces.Box(np.array([0, 0]), high=np.array([WIDTH, HEIGHT]), shape=(2,), dtype=np.int8),
        #     'state': spaces.Discrete(2),
        # })
        #
        # block_blue = spaces.Dict({
        #     'position': spaces.Box(np.array([0, 0]), high=np.array([WIDTH, HEIGHT]), shape=(2,), dtype=np.int8),
        #     'state': spaces.Discrete(2),
        # })
        #
        # self.observation_space = spaces.Dict(
        #     {"gripper": gripper, "cup": cup, "block_red": block_red, "block_blue": block_blue})

        # [ gripper posx, gripper posy, gripper state, cup posx, cup posy, cup state, ...blocks...]
        self.observation_space = spaces.MultiDiscrete(
            [WIDTH, HEIGHT, 2, WIDTH, HEIGHT, 4, WIDTH, HEIGHT, 2, WIDTH, HEIGHT, 2])

        self.step_count = 0
        self.almost_done = 0
        self.wait_for_physics = WAIT_FOR_PHYSICS
        self.max_steps = MAX_STEPS + self.wait_for_physics

        self.screen = None
        self.draw_options = None
        self.reset()

    def reset(self):
        """
        This function resets the environment and returns the game state
        """
        # Move all the objects to their respective starting positions

        self.gripper.body.position = self.gripper.initial_position
        self.cup.body.position = self.cup.initial_position
        self.block1.body.position = self.block1.initial_position
        self.block2.body.position = self.block2.initial_position
        self.cup.body.angle = 0

        self.step_count = 0
        self.almost_done = 0

        # Generate observation
        obs = self._gen_obs()
        return obs

    def render(self, mode='human', close=False):
        """
        This function renders the current game state in the given mode
        """
        FRAME_RATE = 30

        if mode == 'console':
            # print(self._gen_obs())
            clock = pygame.time.Clock()
            clock.tick(FRAME_RATE)
            # self.space.step(1 / 60)
        elif mode == 'human':
            if close:
                pygame.quit()
            else:
                if self.screen is None:
                    pygame.init()
                    self.screen = pygame.display.set_mode(self.size)
                    self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
                self.draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES
                clock = pygame.time.Clock()
                clock.tick(FRAME_RATE)

                self.screen.fill(pygame.color.THECOLORS['white'])

                # Draw gripper
                self.space.debug_draw(self.draw_options)

                pygame.display.flip()

    def step(self, action):
        self.step_count += 1
        reward = 0
        done = False

        if action == self.actions.left:
            try:
                p = self.gripper.body.position
                self.gripper.body.position = (p[0] - self.gripper_step_size, p[1])
            except:
                print("action failed")
        elif action == self.actions.right:
            try:
                p = self.gripper.body.position
                self.gripper.body.position = (p[0] + self.gripper_step_size, p[1])
            except:
                print("action failed")
        elif action == self.actions.up:
            try:
                p = self.gripper.body.position
                self.gripper.body.position = (p[0], p[1] + self.gripper_step_size)
            except:
                print("action failed")
        elif action == self.actions.down:
            try:
                p = self.gripper.body.position
                self.gripper.body.position = (p[0], p[1] - self.gripper_step_size)
            except:
                print("action failed")
        elif action == self.actions.grasp:
            near_items = []
            for item in self.sprite_group:
                if type(item).__name__ == "Block":
                    gap = abs(self.gripper.body.position - item.body.position)
                    if gap[0] < 20 and gap[1] < 20:
                        near_items.append(item)
                elif type(item).__name__ == "Cup":
                    gap = abs(self.gripper.body.position - item.base.body.position)
                    if gap[0] < 20 and gap[1] < 25:
                        near_items.append(item)
            if near_items:
                # print(near_items)
                # sort the near items to find the nearest one
                # near_items.sort(key=lambda x: abs(self.gripper.body.position - x.body.position))
                near_items.sort(key=lambda x: x.body.position.get_dist_sqrd(self.gripper.body.position))
                closest_item = near_items[0]

                # make joints and add them to space
                joint1 = pymunk.constraint.PinJoint(self.gripper.body, closest_item.body, anchor_a=(-20, 10),
                                                    anchor_b=(-15, 0))
                joint2 = pymunk.constraint.PinJoint(self.gripper.body, closest_item.body, anchor_a=(20, 10),
                                                    anchor_b=(15, 0))
                joint3 = pymunk.constraint.PinJoint(self.gripper.body, closest_item.body, anchor_a=(-20, 10),
                                                    anchor_b=(15, 0))
                joint4 = pymunk.constraint.PinJoint(self.gripper.body, closest_item.body, anchor_a=(20, 10),
                                                    anchor_b=(-15, 0))
                self.space.add(joint1, joint2, joint3, joint4)
                self.gripper.grasped = 1

        elif action == self.actions.release:
            joints = []
            if self.space.constraints:
                for constraint in self.space.constraints:
                    joints.append(constraint)
                self.space.remove(joints)
                self.gripper.grasped = 0
        elif action == self.actions.nop:
            pass
        else:
            print("error: action not available")
        time.sleep(self.delay_bw_actions)

        #advance physics
        self.space.step(1 / 60)

        # Get the current observation (i.e., after the physics
        obs = self._gen_obs()

        # Compute rewards
        # reward, done = self._get_reward(obs, done)
        reward, done = self._get_dense_reward(obs, done)

        return obs, reward, done, {}

    def _get_dense_reward(self, obs, done):
        gripx = obs[0]
        gripy = obs[1]
        cupx = obs[3]
        cupy = obs[4]
        blocksx = [obs[6], obs[7]]
        blocksy = [obs[9], obs[10]]

        reward = -0.05
        if self._get_cup_state() == 1:
            self.almost_done += 1  # To ensure that the agent makes sure that a stable state is reached
            if self.almost_done > self.wait_for_physics:
                reward += 1 - .9 * (self.step_count / (self.max_steps))
                done = True
        else:
            if 100 < cupy < 150:
                reward += 0.7
            else:
                if self.gripper.grasped == 1:
                    reward += 0.5
                else:
                    goalb = np.array([cupy+20, cupx])
                    goala = np.array([gripx, gripy])
                    reward = np.linalg.norm(goalb - goala)/200
        if self.step_count > self.max_steps:
            done = True

        return reward, done



    def _get_reward(self, obs, done):
        #Above cup
        gripx = obs[0]
        gripy = obs[1]
        cupx = obs[3]
        cupy = obs[4]
        blocksx = [obs[6], obs[7]]
        blocksy = [obs[9], obs[10]]

        reward = -0.1
        if self.gripper.grasped:
            reward += 1
        if 10 < gripy - cupy < 30:
            reward += 10
        if 0 < gripx - cupx < 10:
            reward += 10
        if gripy > self.size[1]:
            reward -= 1
        if gripy < 20:
            reward -= 1
        if gripx > self.size[0]:
            reward -= 1
        if gripx < 0:
            reward -= 1

        if self._get_cup_state() == 1:
            self.almost_done += 1 #To ensure that the agent makes sure that a stable state is reached
            if self.almost_done > self.wait_for_physics:
                reward += 1 - .9 * (self.step_count / (self.max_steps))
                done = True
        else:
            self.almost_done = 0
        if self.step_count > self.max_steps:
            done = True

        return reward, done


    def _get_cup_state(self):
        raw_angle = abs(self.cup.body.angle * 180 / np.math.pi)
        state = None
        if raw_angle < 45:
            # upside down cup
            state = 3
        elif 45 <= raw_angle < 135:
            # on its side
            if self.cup.body.angle < 0:
                # open left
                state = 4
            else:
                # open right
                state = 2
        elif 135 <= raw_angle < 180:
            # upright
            state = 1
        else:
            state = 1  # for the cases the raw angle is slightly ove r 180

        return state

    def _gen_obs(self):
        """
        Returns current game state
        """
        # obs = {
        #     'gripper': {'position': self.gripper.body.position.int_tuple, 'state': self.gripper.grasped}
        # }
        # # get cup angle
        state = self._get_cup_state()

        # obs['block_blue'] = {'position': np.asarray((0, 0)), 'state': 1}
        # obs['block_red'] = {'position': np.asarray((0, 0)), 'state': 1}

        obs = np.array([self.gripper.body.position.int_tuple[0], self.gripper.body.position.int_tuple[1],
                       self.gripper.grasped,
                       self.cup.body.position.int_tuple[0], self.cup.body.position.int_tuple[1], state,
                       self.block1.body.position.int_tuple[0], self.block1.body.position.int_tuple[1], 0,
                       self.block2.body.position.int_tuple[0], self.block2.body.position.int_tuple[1], 0])

        return obs
