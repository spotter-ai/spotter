from gym_minigrid.minigrid import *
from gym_minigrid.register import register


class SpotterTest(MiniGridEnv):
    """
    Environment with a door and key, sparse reward
    """

    def __init__(self, blocked=False, max_steps=100):
        self.blocked = blocked
        super().__init__(
            grid_size=5,
            max_steps=max_steps
        )
        self.door = None

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        self.grid.wall_rect(0, 0, width, height)
        self.door = Door('yellow', is_locked=False)
        self.put_obj(self.door, 2, width-1)

        # If blocked is true, place a ball directly in front of the door.
        if self.blocked:
            self.put_obj(Ball('blue'), 2, width-2)

        # Place a yellow key on the left side
        # self.place_obj(
        #     Key('yellow'),
        #     top=(1, 5), size=(5, 5)
        # )

        # Place the agent at a random position and orientation in the first room.
        self.place_agent(top=(1, 1), size=(width-2, height-2))

        self.mission = "open the door"

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if action == self.actions.toggle:
            if self.door.is_open:
                reward = self._reward()
                done = True

        return obs, reward, done, info


class UnblockedTest(SpotterTest):
    def __init__(self):
        super().__init__(blocked=False)


class BlockedTest(SpotterTest):
    def __init__(self):
        super().__init__(blocked=True)


register(
    id='MiniGrid-SpotterTestUnblocked-v0',
    entry_point='minigrid_envs.blocked_test:UnblockedTest'
)

register(
    id='MiniGrid-SpotterTestBlocked-v0',
    entry_point='minigrid_envs.blocked_test:BlockedTest'
)
