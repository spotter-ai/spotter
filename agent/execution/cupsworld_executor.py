from gym_cupsworld.envs.wrappers import LastObsWrapper

from agent.execution.executor import Executor
import gym


class CupsWorldExecutor(Executor):
    def __init__(self, env, render_mode, **kwargs):
        self.render_mode = render_mode
        self.env = env
        self.executors = {"pickup": self.pickup,
                          "putdown": self.putdown,
                          "stack": self.stack,
                          "unstack": self.unstack,
                          "cover": self.cover,
                          "dropin": self.dropin,
                          "flipup": self.flipup,
                          "go_above": self.go_above,
                          "go_to": self.go_to,
                          "go_to_table": self.go_to_table,
                          "lift": self.lift,
                          "lower": self.lower,
                          "right": self.right,
                          "left": self.left,
                          "up": self.up,
                          "down": self.down,
                          "grasp": self.grasp,
                          "release": self.release}

    def _all_actions(self):
        raise NotImplementedError

    def _act(self, operator_name):
        executor_name, items = self._operator_parts(operator_name)
        executor = self.executors[executor_name]
        if items:
            # param operator call  E.g. "(stack block cup)"
            return executor(items)
        else:
            # primitive action call E.g., "right"
            return executor()
        # return True

    ####################################
    # OPERATORS
    ###################################

    def pickup(self, items):
        if len(items) == 1:
            #go above item
            self.go_to(items)
            # grab item
            self.grasp()
            # lift it up
            self.lift()
            return self.wait()
        else:
            raise Exception("Agent cannot pickup multiple objects")

    def putdown(self, items=[]):
        if len(items) <= 1:
            self.go_to_table()
            return self.release()
        else:
            raise Exception("Cannot putdown multiple specified objects")

    def stack(self, items):
        """
        stack first item on top of the second item
        -assumes you are already holding first item
        """
        # move above second item
        dest = []
        dest.append(items[1])
        self.go_above(dest)
        self.wait()
        self.release()
        return self.wait()

    def unstack(self, items):
        """
        unstack first item FROM the top of second item
        """
        dest = []
        dest.append(items[0])
        self.go_to(dest)
        self.grasp()
        self.lift()
        return self.wait()

    def cover(self, items):
        """
        Cover the first item (which is a cup) on the second
        """
        return self.stack(items)

    def dropin(self, items):
        """
        Drop in the first item (block) into the second item (cup)
        """
        return self.stack(items)

    def flipup(self, items):
        """
        flip the cup
        """
        if len(items) == 1:
            self.pickup(items)
            self.go_above(["block_red"])
            self.go_to_relative(xoff=10)
            self.release()
            return self.wait(50)

    def go_above(self, items):
        """
        Move gripper to well above the object (useful when carrying something to
        stack
        E.g., item = "block_blue"
        """
        if len(items) == 1:
            offset = 100
            return self.go_to(items, offset)
        else:
            raise Exception("Cannot go above more than one item")

    def go_to(self, items, offset=10):
        """
        Move gripper to a graspable location just above an object
        """
        if len(items) == 1:
            obs = self.env.last_observation()
            # find position above item
            if items[0] =='block_blue':
                x = obs[6]
                y = obs[7]
            elif items[0] == 'block_red':
                x = obs[9]
                y = obs[10]
            elif items[0] == 'cup':
                x = obs[3]
                y = obs[4]
                y = y + 12
            else:
                raise Exception("Sorry! Unknown item")
            return self._go_to_xy(x, y+offset)
        else:
            raise Exception("Cannot go to more than one item")

    def go_to_table(self, items=None):
        """
        Goes to a location above an empty table spot
        """
        window = 80
        obs = self.env.last_observation()
        xpos = [obs[3], obs[6], obs[9]]
        ypos = [obs[4], obs[7], obs[10]]
        blocked = []
        # is the ypos of any object below a certain height
        for idx,j in enumerate(ypos):
            if j < 50:
                # this is a low object
                blocked.append((xpos[idx]-15, xpos[idx]+15))
        blocked.sort(key=lambda tup: tup[0])
        open = []
        for idx, range in enumerate(blocked):
            if idx < len(blocked)-1:
                open.append((blocked[idx][1], blocked[idx+1][0]))
        open.append( (0, blocked[0][0]) )
        open.append( (blocked[-1][1], 200) )
        open.sort(key=lambda tup: tup[0])

        max_width = 0
        best_x = None
        for entry in open:
            width = entry[1] - entry[0]
            if width > max_width:
                max_width = width
                best_x = (entry[0] + entry[1])//2

        if best_x:
            return self._go_to_xy(best_x, 100)
        else:
            return

    def lift(self, items=None):
        if items:
            self.go_to(items)
            return self.lift()
        else:
            for _ in range(22):
                self.up()
            return self.wait()

    def lower(self, items=None):
        for _ in range(40):
            self.down()
        return self.wait()

    def go_to_relative(self, xoff=0, yoff=0):
        obs = self.env.last_observation()
        x = obs[0] + xoff
        y = obs[1] + yoff
        return self._go_to_xy(x, y)

    def wait(self, amount=30):
        for _ in range(amount):
            self.nop()
        return self.nop()

    ###################################################
    # ACTIONS
    ##################################################

    def execute_core_action(self, action):
        return self.env.step(action)

    def right(self):
        if self.render_mode == 'HUMAN':
            self.env.render()
        else:
            self.env.render('console')
        return self.env.step(self.env.actions.right)

    def left(self):
        if self.render_mode == 'HUMAN':
            self.env.render()
        else:
            self.env.render('console')
        return self.env.step(self.env.actions.left)

    def up(self):
        if self.render_mode == 'HUMAN':
            self.env.render()
        else:
            self.env.render('console')
        return self.env.step(self.env.actions.up)

    def down(self):
        if self.render_mode == 'HUMAN':
            self.env.render()
        else:
            self.env.render('console')
        self.env.step(self.env.actions.nop)
        return self.env.step(self.env.actions.down)

    def grasp(self):
        if self.render_mode == 'HUMAN':
            self.env.render()
        else:
            self.env.render('console')
        return self.env.step(self.env.actions.grasp)

    def release(self):
        if self.render_mode == 'HUMAN':
            self.env.render()
        else:
            self.env.render('console')
        return self.env.step(self.env.actions.release)

    def nop(self):
        if self.render_mode == 'HUMAN':
            self.env.render()
        else:
            self.env.render('console')
        return self.env.step(self.env.actions.nop)

    #############################
    # Utilities
    ############################

    def _operator_parts(self, operator_name):
        """
        input: "(pickup cup)"
        output: "pickup", ["cup"]
        """
        rem_parens = operator_name.replace("(","").replace(")","")
        return rem_parens.split(" ")[0], rem_parens.split(" ")[1:]

    def _go_to_xy(self, x, y):
        """
        Go to specific x, y position
        """
        obs = self.env.last_observation()
        gripper_x = obs[0]
        gripper_y = obs[1]

        if y > gripper_y:
            for _ in range((y - gripper_y)//2):
                self.up()
            if x > gripper_x:
                for _ in range((x - gripper_x)//2):
                    self.right()
            else:
                for _ in range(abs(x - gripper_x)//2):
                    self.left()
        else:
            for _ in range((gripper_y - y)//2):
                self.down()
            if x > gripper_x:
                for _ in range((x - gripper_x)//2):
                    self.right()
            else:
                for _ in range(abs(x - gripper_x)//2):
                    self.left()

        obs = self.env.last_observation()
        return obs


if __name__ == "__main__":
    env = gym.make('CupsWorld-v0')
    env = LastObsWrapper(env)
    obs = env.reset()
    executor = CupsWorldExecutor(env, render_mode='HUMAN')
    obs, reward, done, info = executor.act("(right)")
    # print(obs,reward, done, info)
    obs, reward, done, info = executor.act("(pickup block_blue")
    # print("after pickup: ", obs, reward, done, info)
    obs, reward, done, info = executor.act("(stack block_blue block_red)")
    # print("after stack: ", obs, reward, done, info)
    executor.act("(pickup cup")
    executor.act("(flipup cup)")
    obs, reward, done, info = executor.act("(unstack block_blue block_red)")
    # print("after unstack: ", obs, reward, done, info)
    executor.act("(dropin block_blue cup)")
    executor.wait(50)











