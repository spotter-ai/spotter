from gym_minigrid.minigrid import *
from gym_minigrid.register import register


class TwoRoom(MiniGridEnv):
    """
    Environment with a door and key, sparse reward
    """

    def __init__(self, blocked=False, need_goal=True, max_steps=100):
        self.blocked = blocked
        super().__init__(
            grid_size=16,
            max_steps=max_steps
        )
        self.door = None
        self.need_goal = need_goal

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        self.grid.wall_rect(0, 4, 7, 7)
        self.grid.wall_rect(6, 0, 10, 16)
        # Generate the surrounding walls
        # self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)
        self.goal_pos = [width-2, height-2]

        self.door =Door('yellow', is_locked=True)
        self.put_obj(self.door, 6, 7)
        # Place a door in the wall
        # doorIdx = self._rand_int(1, width-2)
        # self.put_obj(Door('yellow', is_locked=True), splitIdx, doorIdx)

        # If blocked is true, place a ball directly in front of the door.
        if self.blocked:
            self.put_obj(Ball('blue'), 5, 7)

        # Place a yellow key on the left side
        self.place_obj(
            Key('yellow'),
            top=(1,5), size=(5,5)
        )

        # Place the agent at a random position and orientation in the first room.
        self.place_agent(top=(1, 5), size=(5, 5))



        self.mission = "use the key to open the door and then get to the goal"
    
    def step(self, action):
        obs, reward, done, info = super().step(action)

        if not self.need_goal and action == self.actions.toggle:
            if self.door.is_open:
                reward = self._reward()
                done = True

        return obs, reward, done, info


class SpotterLevel1(TwoRoom):
    def __init__(self):
        super().__init__(blocked=False, need_goal=False)


class SpotterLevel2(TwoRoom):
    def __init__(self):
        super().__init__(blocked=True, need_goal=False)


class SpotterLevel3(TwoRoom):
    def __init__(self):
        super().__init__(blocked=True, need_goal=True, max_steps=250)


class UnblockedGoal(TwoRoom):
    def __init__(self):
        super().__init__(blocked=False, need_goal=True, max_steps=250)


register(
    id='MiniGrid-UnblockedGoal-v0',
    entry_point='minigrid_envs.blocked_two_room:UnblockedGoal'
)


register(
    id='MiniGrid-TwoRooms-v0',
    entry_point='minigrid_envs.blocked_two_room:TwoRoom'
)

register(
    id='MiniGrid-SpotterLevel1-v0',
    entry_point='minigrid_envs.blocked_two_room:SpotterLevel1'
)

register(
    id='MiniGrid-SpotterLevel2-v0',
    entry_point='minigrid_envs.blocked_two_room:SpotterLevel2'
)

register(
    id='MiniGrid-SpotterLevel3-v0',
    entry_point='minigrid_envs.blocked_two_room:SpotterLevel3'
)
