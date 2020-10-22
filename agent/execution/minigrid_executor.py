import math
from collections import defaultdict
import heapq

from agent.execution.executor import Executor, operator_parts, RewardAccumulator
from itertools import product
from gym_minigrid.window import Window
import matplotlib.pyplot as plt

VEC_TO_DIR = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}
DIR_TO_VEC = [(1, 0), (0, 1), (-1, 0), (0, -1)]

SAVE_VIDEO = True

class MiniGridExecutor(Executor):

    def _remove_learner(self, learner):
        self._learners.remove(learner)

    def __init__(self, env, detector=None, render_mode=None, learners=[], **kwargs):
        self.env = env
        self.core_actions = self.env.actions
        self.detector = detector
        self._learners = learners
        self.executors = {"gotoobj1": (["agent", "graspable"], self.go_to_obj),
                          "gotoobj2": (["agent", "notgraspable"], self.go_to_obj),
                          "gotoobj3": (["agent", "graspable", "thing"], self.go_to_obj),
                          "gotoobj4": (["agent", "notgraspable", "thing"], self.go_to_obj),
                          "gotodoor1": (["agent", "door"], self.go_to_obj),
                          "gotodoor2": (["agent", "door", "thing"], self.go_to_obj),
                          "stepinto": (["agent", "goal"], self.step_into),
                          "enterroomof": (["agent", "door", "physobject"], self.enter_room_of),
                          "usekey": (["agent", "key", "door"], self.usekey),
                          "opendoor": (["agent", "door"], self.usekey),
                          "pickup": (["agent", "graspable"], self.pickup),
                          "putdown": (["agent", "graspable"], self.putdown),
                          "core_drop": ([], self.core_drop),
                          "core_pickup": ([], self.core_pickup),
                          "core_forward": ([], self.core_forward),
                          "core_right": ([], self.core_right),
                          "core_left": ([], self.core_left),
                          "core_toggle": ([], self.core_toggle),
                          "core_done": ([], self.core_done)}
        self._window = None
        self._record_path = None
        self.set_record_path(0)
        if render_mode and render_mode.lower() == "human":
            self._window = Window("minigrid_executor")
            self._window.show(block=False)
        self.accumulator = None

    def _attach_learner(self, learner):
        self._learners.append(learner)

    def _clear_learners(self):
        self._learners = []

    def executor_map(self):
        return {name: (self, self.executors[name][0], self.executors[name][1]) for name in self.executors}

    def _all_actions(self):
        action_list = []
        type_to_objects = self.detector.get_objects(inherit=True)
        for executor_name in self.executors:
            possible_args = tuple([type_to_objects[t] for t in self.executors[executor_name][0]])
            for items in product(*possible_args):
                if len(items) == 0:
                    action_list.append("(" + executor_name + ")")
                else:
                    action_list.append("(" + executor_name + " " + " ".join(items) + ")")
        return action_list

    def _act(self, operator_name):
        self.accumulator = None
        executor_name, items = operator_parts(operator_name)
        param_types, executor = self.executors[executor_name]
        if len(items) == len(param_types):
            if items:
                # Doesn't currently ensure actions are typesafe. Could if we wanted it to.
                # param operator call  E.g. "(stack block cup)"
                obs, reward, done, info = executor(items)
            else:
                # primitive action call E.g., "right"
                obs, reward, done, info = executor()
        else:
            raise Exception("Wrong number of arguments for action " + executor_name)
        return obs, reward, done, info

    ####################################
    ## OPERATORS
    ######################

    def _check_agent_nextto(self, item, obs):
        return self.detector.check_nextto("agent", item, obs)

    def _check_agent_facing(self, item, obs):
        return self.detector.check_facing("agent", item, obs)

    def _check_agent_obstructed(self, obs):
        return self.detector.check_formula(["(obstructed agent)"], obs)

    def get_line_between(self, x1, y1, x2, y2):
        # Assumes a straight line, so only one of x2-x1, y2-y1 is nonzero
        if y1 == y2:
            return [(i, y1) for i in range(min(x1, x2) + 1, max(x1, x2))]
        else:
            return [(x1, i) for i in range(min(y1, y2) + 1, max(y1, y2))]

    def enter_room_of(self, items):
        self.accumulator = RewardAccumulator(self.env)
        try:
            if self.detector.check_formula(["(inroom {} {})".format(items[0], items[2])], self.env.last_observation()):
                self.execute_core_action(self.env.actions.left, self.accumulator)
                self.execute_core_action(self.env.actions.left, self.accumulator)
            else:
                self.execute_core_action(self.env.actions.forward, self.accumulator)
                self.execute_core_action(self.env.actions.forward, self.accumulator)
        finally:
            return self.accumulator.combine_steps()

    def go_to_obj(self, items):
        self.accumulator = RewardAccumulator(self.env, action_timeout=100)
        agent = items[0]
        obj = items[1]
        obs = self.env.last_observation()
        agent_cur_direction = obs['objects']['agent']['encoding'][2]
        agent_x = obs['objects'][agent]['x']
        agent_y = obs['objects'][agent]['y']
        object_x = obs['objects'][obj]['x']
        object_y = obs['objects'][obj]['y']
        goal = (object_x, object_y)
        initial_state = (agent_x, agent_y, agent_cur_direction)
        image = self.env.last_observation()['image']
        path = self.a_star(image, initial_state, goal, self.manhattan_distance)
        try:
            for action in path[:-1]:
                self.execute_core_action(action, self.accumulator)
        finally:
            return self.accumulator.combine_steps()

    def manhattan_distance(self, agent_orientation, goal_position):
        return abs(agent_orientation[0] - goal_position[0]) + abs(agent_orientation[1]-goal_position[1])

    def get_path(self, parent, current):
        plan = []
        while current in parent.keys():
            current, action = parent[current]
            plan.insert(0, action)
        return plan

    def a_star(self, image, initial_state, goal, heuristic):
        open_set = []
        parent = {}
        g_score = defaultdict(lambda: math.inf)
        g_score[initial_state] = 0.
        f_score = defaultdict(lambda: math.inf)
        f_score[initial_state] = heuristic(initial_state, goal)
        heapq.heappush(open_set, (f_score[initial_state], initial_state))

        while len(open_set) > 0:
            f, current = heapq.heappop(open_set)
            if (current[0], current[1]) == goal:
                return self.get_path(parent, current)
            neighbors = [(current[0], current[1], (current[2] - 1) % 4),
                         (current[0], current[1], (current[2] + 1) % 4)]
            fwd_x = current[0] + DIR_TO_VEC[current[2]][0]
            fwd_y = current[1] + DIR_TO_VEC[current[2]][1]
            if image[fwd_x][fwd_y][0] in [1, 3, 8] or (fwd_x, fwd_y) == goal:
                neighbors.append((fwd_x, fwd_y, current[2]))
            for action, neighbor in enumerate(neighbors):
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score[neighbor]:
                    parent[neighbor] = (current, action)
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                    if neighbor not in [x[1] for x in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return []

    def set_record_path(self, episode_number):
        self._record_path = "results/videos/{}/episode_{}".format(self.env.spec.id, episode_number)
        if SAVE_VIDEO:
            import os
            os.makedirs(self._record_path, exist_ok=True)

    def pickup(self, items):
        return self.core_pickup()

    def putdown(self, items):
        return self.core_drop()

    def usekey(self, items):
        return self.core_toggle()

    def step_into(self, items):
        return self.core_forward()

    def _execute_core_action(self, action, accumulator=None):
        prev_obs = self.env.last_observation()
        obs, reward, done, info = self.env.step(action)
        for learner in self._learners:
            learner.train(prev_obs, action, reward, obs)
        if accumulator:
            accumulator.accumulate(action, obs, reward, done, info)
        if self._window:
            img = self.env.render('rgb_array', tile_size=32, highlight=False)
            self._window.show_img(img)
            if SAVE_VIDEO:
                self._window.fig.savefig(self._record_path + "/{}.png".format(self.env.step_count))
        return obs, reward, done, action

    def core_drop(self):
        return self._execute_core_action(self.env.actions.drop)

    def core_pickup(self):
        return self._execute_core_action(self.env.actions.pickup)

    def core_forward(self):
        return self._execute_core_action(self.env.actions.forward)

    def core_left(self):
        return self._execute_core_action(self.env.actions.left)

    def core_right(self):
        return self._execute_core_action(self.env.actions.right)

    def core_toggle(self):
        return self._execute_core_action(self.env.actions.toggle)

    def core_done(self):
        return self._execute_core_action(self.env.actions.done)
