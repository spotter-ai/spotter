import itertools
from typing import List

from agent.detection.detector import Detector
from agent.statehashing.minigrid_hashing import MinigridStateHasher
from representation.ow_pddl.parser import Parser

OBJECT_TYPE_LIST = [
'unseen',
'empty',
'wall',
'floor',
'door',
'key',
'ball',
'box',
'goal',
'lava',
'agent'
]


class MiniGridDetector(Detector):

    def __init__(self, env, domain=None):
        super().__init__()
        self.domain = domain
        self.state = None
        self.objects = None
        self.env = env
        if self.env.last_observation() is None:
            self.env.reset() # NB this is risky, but we need it bc in minigrid we don't know the objects until we have a
        #                      first state.

        self.domain = Parser(domain).parse_domain()

        self.detectors = {"closed": (["door"], self.closed_detector),
                          "locked": (["door"], self.locked_detector),
                          "open": (["door"], self.open_detector),
                          "inroom": (["agent", "physobject"], self.inroom_detector),
                          "blocked": (["door"], self.blocked_detector),
                          "handsfree": (["agent"], self.handsfree_detector),
                          "obstructed": (["agent"], self.obstructed_detector),
                          "nexttofacing": (["agent", "thing"], self.nexttofacing_detector),
                          "holding": (["agent", "graspable"], self.holding_detector),
                          "atgoal": (["agent", "goal"], self.atgoal_detector),
                          }

        self._hasher = MinigridStateHasher(self)
        self._fluents_list = None
        self._cached_state = None
        self._cached_fs = None

    def _predicate_typesafe(self, pred_name, param_types):
        signature = self.domain.predicates[pred_name].signature
        if len(param_types) != len(signature):
            return False
        for i, param_type in enumerate(param_types):
            while signature[i][1][0].name != param_type:
                parent = self.domain.types[param_type].parent
                if parent is None:
                    return False
                param_type = parent.name
        return True

    def _all_fluents(self):
        if self._fluents_list is None:
            state = self.env.last_observation()

            self.objects = frozenset(state['objects'].keys())

            type_to_objects = self._get_objects(inherit=True)

            fluent_state = set()

            for detector_name in self.detectors:
                possible_args = tuple([type_to_objects[t] if t in type_to_objects else []
                                       for t in self.detectors[detector_name][0]])
                for items in itertools.product(*possible_args):
                    if len(items) == 0:
                        fluent_state.add("(" + detector_name + ")")
                    else:
                        fluent_state.add("(" + detector_name + " " + " ".join(items) + ")")
            self._fluents_list = list(fluent_state)
        return self._fluents_list

    def _get_object_types(self, obs):
        return {obj: OBJECT_TYPE_LIST[obs['objects'][obj]['encoding'][0]] for obj in obs['objects'].keys()}

    def _get_objects(self, inherit=False):
        obs = self.env.last_observation()
        state_objects = obs['objects'].keys()
        type_to_objs_dict = {}
        for obj in state_objects:
            obj_typename = OBJECT_TYPE_LIST[obs['objects'][obj]['encoding'][0]]
            if inherit:
                obj_type = self.domain.types[obj_typename]
                while obj_type is not None:
                    type_to_objs_dict.setdefault(obj_type.name, []).append(obj)
                    obj_type = obj_type.parent
            else:
                type_to_objs_dict.setdefault(obj_typename, []).append(obj)
        return type_to_objs_dict

    def _detect(self):
        self.state = self.env.last_observation()
        return self._interpret(self.state)

    @staticmethod
    def _direction(state):
        if 'direction' in state:
            return state['direction']
        else:
            return state['objects']['agent']['encoding'][2]

    def _interpret(self, state):
        if self._cached_state is None or self._hasher.hash(self._cached_state) != self._hasher.hash(state):
            self._cached_state = state
            self.objects = frozenset(state['objects'].keys())

            type_to_objects = self._get_objects(inherit=True)

            fluent_state = set()

            for detector_name in self.detectors:

                possible_args = tuple([type_to_objects[t] if t in type_to_objects else []
                                       for t in self.detectors[detector_name][0]])
                for items in itertools.product(*possible_args):
                    if len(items) == 0:
                        if self.detectors[detector_name][1](state):
                            fluent_state.add("(" + detector_name + ")")
                    else:
                        if self.detectors[detector_name][1](*items, state):
                            fluent_state.add("(" + detector_name + " " + " ".join(items) + ")")
            self._cached_fs = list(fluent_state)
        return self._cached_fs

    # This blocked detector is only "approximately" accurate, but it works for our experiments.
    def blocked_detector(self, item, obs):
        # Currently the blocked predicate is always false for anything that's not a door.
        if obs['objects'][item]['encoding'][0] != 4:
            return False
        door = item
        types_to_objects = self._get_objects(inherit=True)
        for obj in types_to_objects['graspable']:
            if self.check_nextto(obj, door, obs):
                return True
        return False

    def inroom_detector(self, agent, item, obs):
        if self.holding_detector(agent, item, obs):
            return True
        if self._x(item, obs) == self._x("door", obs):
            return True
        else:
            return (self._x(agent, obs) > self._x("door", obs)) == (self._x(item, obs) > self._x("door", obs))

    def atgoal_detector(self, agent, goal, obs):
        return self._x(agent, obs) == self._x(goal, obs) and self._y(agent, obs) == self._y(goal, obs)

    def closed_detector(self, item, obs):
        return self._closed(item, obs)

    def locked_detector(self, item, obs):
        return self._locked(item, obs)

    def open_detector(self, item, obs):
        return self._open(item, obs)

    def nexttofacing_detector(self, item1, item2, obs):
        return self.check_nextto(item1, item2, obs) and self.check_facing(item1, item2, obs)

    def check_nextto(self, obj1, obj2, obs):
        object1_x = self._x(obj1, obs)
        object2_x = self._x(obj2, obs)
        object1_y = self._y(obj1, obs)
        object2_y = self._y(obj2, obs)
        return object2_x != -1 and object2_y != -1 and abs(object2_x - object1_x) + abs(object2_y - object1_y) == 1

    def check_facing(self, agent, obj, obs):
        agent_x = self._x(agent, obs)
        object_x = self._x(obj, obs)
        agent_y = self._y(agent, obs)
        object_y = self._y(obj, obs)
        direction = self._direction(obs)
        return (object_x > agent_x and direction == 0) or (object_x < agent_x and direction == 2) \
               or (object_y > agent_y and direction == 1) or (object_y < agent_y and direction == 3)

    def obstructed_detector(self, agent, obs):
        if not self._is_agent(agent, obs):
            return False
        for obj in self.objects:
            if self.nexttofacing_detector(agent, obj, obs):
                return True
        return False

    def notobstructed_detector(self, agent, obs):
        if not self._is_agent(agent, obs):
            return False
        return not self.obstructed_detector(agent, obs)

    @staticmethod
    def _is_agent(item, obs):
        return obs['objects'][item]['encoding'][0] == 10

    @staticmethod
    def _x(item, obs):
        return obs['objects'][item]['x']

    @staticmethod
    def _y(item, obs):
        return obs['objects'][item]['y']

    @staticmethod
    def _closed(item, obs):
        return obs['objects'][item]['encoding'][2] == 1

    @staticmethod
    def _locked(item, obs):
        return obs['objects'][item]['encoding'][2] == 2

    @staticmethod
    def _open(item, obs):
        return obs['objects'][item]['encoding'][2] == 0

    def holding_detector(self, item1, item2, obs):
        """
        Check if the agent (item1) is holding item2.
        """
        return self._is_agent(item1, obs) and self._x(item2, obs) == -1 and \
               self._y(item2, obs) == -1

    def notholding_detector(self, item1, item2, obs):
        return self._is_agent(item1, obs) and self._x(item2, obs) != -1

    def handsfree_detector(self, item1, obs):
        """
        True if the agent (specified by item1) isn't holding anything.
        """
        if not self._is_agent(item1, obs):
            return False
        for obj in self.objects:
            if self.holding_detector(item1, obj, obs):
                return False
        return True

    def _fluent_name(self, fluent):
        return fluent[1:-1].split(' ')[0]

    def _fluent_args(self, fluent):
        return fluent[1:-1].split(' ')[1:]

    def _check_formula(self, formula: List[str], state):
        """
        check if conjunctive formula of fluents holds in current state
        - to check preconditions and goals and effects
        """

        self.objects = frozenset(state['objects'].keys())

        object_types = self._get_object_types(state)

        for fluent in formula:
            fname = self._fluent_name(fluent)
            fargs = self._fluent_args(fluent)
            if not self._predicate_typesafe(fname, [object_types[arg] for arg in fargs]):
                return False
            if not self.detectors[fname][1](*fargs, state):
                return False
        return True

    def _check_domain(self):
        """
        Make sure that each predicate in the domain file has a corresponding low level detector
        """
        raise NotImplementedError
