import itertools
from typing import List
import gym
from gym_cupsworld.envs.wrappers import LastObsWrapper

from agent.detection.detector import Detector


class CupsWorldDetector(Detector):
    def __init__(self, env, domain=None):
        self.env = env
        self.domain = domain
        self.objects = ["gripper", "cup", "block_blue", "block_red"]
        self.object_type_map = {"block": ["block_blue", "block_red"], "cup": ["cup"]}
        self.zero_ary_detectors = [self.handempty_detector]
        self.one_ary_detectors = [self.ontable_detector,
                          self.clear_detector,
                          self.holding_detector,
                          self.facedown_detector,
                          self.upright_detector,
                          self.empty_detector]
        self.two_ary_detectors = [self.on_detector, self.inside_detector]
        self.state = None
        self.pos = None

    def _get_objects(self, options):
        return self.object_type_map

    def _detect(self):
        state = self.env.last_observation()
        return self._interpret(state)

    def _all_fluents(self):
        pass

    def _interpret(self, state):
        """
        input: state = [ gx, gy, gs, cx, cy, cs, bbx, bby, bbs, brx, bry brs]
        E.g., return ["(ontable blk1)", "(ontable blk2)", "(ontable c1)", "(upright c1)"]
        """
        self.state = state
        self.pos = {"gripper": (state[0], state[1]),
                    "cup": (state[3], state[4]),
                    "block_blue": (state[6], state[7]),
                    "block_red": (state[9], state[10])}
        fluent_state = set()
        for detector in self.zero_ary_detectors:
            (fluent, valuation) = detector()
            if valuation:
                fluent_state.add(fluent)
        for detector in self.one_ary_detectors:
            for item in self.objects:
                (fluent, valuation) = detector(item)
                if valuation:
                    fluent_state.add(fluent)

        for detector in self.two_ary_detectors:
            for i, j in itertools.product(self.objects, self.objects):
                if i != j:
                    (fluent, valuation) = detector(i, j)
                    if valuation:
                        fluent_state.add(fluent)

        return list(fluent_state)

    def ontable_detector(self, item):
        """
        ontable(item) true if item is below the height of 40
        """
        fluent = "(ontable "+item+")"
        if "gripper" in item:
            valuation = False
        else:
            if self.pos[item][1] < 40:
                valuation = True
            else:
                valuation = False
        return fluent, valuation

    def clear_detector(self, item):
        """
        clear(item) true if item has nothing immediately above it
        """
        fluent = "(clear "+item+")"

        valuation = False
        if item != 'gripper':
            for key, value in self.pos.items():
                if key != item and key != 'gripper':
                    if abs(self.pos[key][0] - self.pos[item][0]) < 30:
                        if 10< self.pos[key][1] - self.pos[item][1] < 50:
                            valuation = False
                        else:
                            valuation = True
                    else:
                        valuation = True

        return fluent, valuation

    def handempty_detector(self):
        """
        Check if gripper not grasping anything
        """
        fluent = "(handempty)"
        valuation = False

        grasped = self.state[2]
        if grasped == 0:
            valuation = True
        return fluent, valuation

    def holding_detector(self, item):
        fluent = "(holding "+item+")"
        valuation = False

        # detect
        # holding = Gripper grasped object and object not on the table
        if "gripper" not in item:
            _, handempty = self.handempty_detector()
            _, ontable = self.ontable_detector(item)
            if not handempty and not ontable:
                valuation = True
        return fluent, valuation

    def facedown_detector(self, item):
        fluent = "(facedown "+item+")"
        valuation = False

        if item == 'cup':
            orientation = self.state[5]
            if orientation == 3:
                valuation = True

        return fluent, valuation

    def upright_detector(self, item):
        fluent = "(upright " + item + ")"
        valuation = False

        # detect

        if item == 'cup':
            orientation = self.state[5]
            if orientation == 1:
                valuation = True
        return fluent, valuation

    def empty_detector(self, item):
        fluent = "(empty " + item + ")"
        valuation = False

        # if there exists any object that is within threshold x and y
        if item == 'cup':
            valuation = True
            for key, value in self.pos.items():
                if key != item and key != 'gripper':
                    if abs(self.pos[key][0] - self.pos[item][0]) < 10:
                        if abs(self.pos[key][1] - self.pos[item][1]) < 10:
                            valuation = False

        return fluent, valuation

    def on_detector(self, item1, item2):
        fluent = "(on "+ item1 + " " + item2 + ")"
        valuation = False

        #detect
        if "gripper" not in item1 and "gripper" not in item2:
            if self.pos[item1][1] > self.pos[item2][1]:
                if abs(self.pos[item1][0] - self.pos[item2][0]) < 30:
                    if self.pos[item1][1] - self.pos[item2][1] < 50:
                        valuation = True

        return fluent, valuation

    def inside_detector(self, item1, item2):
        fluent = "(inside " + item1 + " " + item2 + ")"
        valuation = False

        # detect
        if "block" in item1 and "cup" in item2:
            if abs(self.pos[item1][0] - self.pos[item2][0]) < 10:
                if abs(self.pos[item1][1] - self.pos[item2][1]) < 10:
                    valuation = True

        return fluent, valuation

    def _check_formula(self, formula: List[str], state):
        """
        check if conjunctive formula of fluents holds in current state
        - to check preconditions and goals and effects
        """
        fluent_state = self._interpret(state)
        if set(formula) .issubset(set(fluent_state)):
            return True
        else:
            return False

    def _check_domain(self):
        """
        Make sure that each predicate in the domain file has a corresponding low level detector
        """
        raise NotImplementedError


if __name__ == "__main__":
    env = gym.make('CupsWorld-v0')
    env = LastObsWrapper(env)
    obs = env.reset()
    # state_start = [68,  75,   0, 133,  37,   3,  25,  37,   0,  66,  37,   0]
    # state_cover = [136,  59,   1, 120,  34,   3,  25,  29,   0, 119,  29,   0]
    # state_stack = [32, 107,   0, 133,  36,   3,  24,  29,   0,  24,  56,   0]
    state_holding = [26,  91,   1, 133,  36,   3,  24,  72,   0,  66,  29,   0]
    # formula1 = ['(on block_red block_blue)']
    # formula2 = ['(clear block_blue)']
    formula3 = ['(holding block_blue)']
    detector = CupsWorldDetector(env=env)
    fluent_state = detector.interpret(state_holding)
    print(fluent_state)

    # assert detector.check_formula(formula1, state_stack) is True
    # assert detector.check_formula(formula2, state_stack) is False
    assert detector.check_formula(formula3, state_holding) is True




