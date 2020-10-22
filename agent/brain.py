# Create and configure logger
import logging
import math
from typing import List, Tuple, Dict

from agent.env_wrappers import LastObsWrapper
from agent.execution.extendable_executor import ExtendableExecutor

from representation import grounding
from representation.ow_pddl.parser import Parser
from representation.ow_pddl.ow_pddl import Problem, Predicate
from representation.task import Task, Operator

import gym

from helper import get_root_dir
logging.basicConfig(filename=get_root_dir() + "/logs/console.log",
                    format="%(name)s — %(levelname)s — %(message)s",
                    filemode='w')

# Creating an object
log = logging.getLogger("brain")

# Setting the threshold of logger to DEBUG
log.setLevel(logging.DEBUG)

MAX_EPSILON = 0.90
MIN_EPSILON = 0.05
EXPLORATION_STOP = 1000000
LAMBDA = -math.log(0.01) / EXPLORATION_STOP # speed of decay

class Affect:
    """
    Class instantiated in Brain for capturing various affective states of the agent.
    """

    def __init__(self):
        self._giveup = False
        self._blocked = False
        self._planning_impasse = False
        self._execution_impasse = False

    @property
    def giveup(self) -> bool:
        return self._giveup

    @giveup.setter
    def giveup(self, value: bool):
        self._giveup = value

    @property
    def blocked(self) -> bool:
        return self._blocked

    @blocked.setter
    def blocked(self, value: bool):
        self._blocked = value

    @property
    def planning_impasse(self) -> bool:
        return self._planning_impasse

    @planning_impasse.setter
    def planning_impasse(self, value: bool):
        self._planning_impasse = value

    @property
    def execution_impasse(self) -> bool:
        return self._execution_impasse

    @execution_impasse.setter
    def execution_impasse(self, value: bool):
        self._execution_impasse = value


class Configuration:
    """
    Class instantiated in Brain for various configuration settings for the agent
    """

    def __init__(self, creativity: str = "macgyver"):
        self._creativity = creativity

    @property
    def creativity(self) -> str:
        return self._creativity

    @creativity.setter
    def creativity(self, creativity: str = "macgyver"):
        self._creativity = creativity


class Motor:
    """
    Class instantiated in Brain for executing actions
    """

    def __init__(self, env, domain_bias, detector, executor_class, state_hasher=None, executor=None, render=None):
        self.env = env
        self.env.reset()
        self.domain = domain_bias
        self.detector = detector(env=self.env, domain=domain_bias)
        self._num_eps_updates = 0
        self._state_hasher = state_hasher(detector=self.detector)
        if executor is None:
            self.executor = ExtendableExecutor(env=self.env, detector=self.detector,
                                               base_executor=executor_class(env=self.env,
                                                                      detector=self.detector, render_mode=render))
        else:
            self.executor = executor

    @property
    def epsilon(self):
        return MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) *\
                      math.exp(-LAMBDA*self._num_eps_updates)

    @property
    def state_hasher(self):
        return self._state_hasher

    def increment_episode(self):
        self._num_eps_updates += 1
        if self.executor is not None:
            self.executor.set_record_path(self._num_eps_updates)

    @property
    def env(self):
        return self._env

    def reset(self):
        return self.env.reset()

    def get_initial_problem(self, goal):
        """
        Constructs a pddl "Problem" file
        using goal provided.

        Object:
        """
        domain_name_str = self._generate_domain_name()
        objects_str = self._generate_objects_line()
        init_str = self._generate_init_str()
        goal_str = goal

        # make the problem file
        name = None
        with open("domains/problem.pddl", "w") as f:
            f.write("(define (problem "+domain_name_str+"-0-1)")
            f.write("\n")
            f.write("(:domain "+domain_name_str+")")
            f.write("\n")
            f.write("(:objects "+objects_str+")")
            f.write("\n")
            f.write("(:INIT "+init_str+")")
            f.write("\n")
            f.write("(:goal "+goal_str+")")
            f.write("\n")
            f.write(")")
            name = f.name

        return name

    def _generate_domain_name(self):
        # read self.domain and get domain name
        with open(self.domain, "r") as f:
            for line in f.readlines():
                if "define" in line:
                    domain_name = line.split(" ")[2].replace(")","").replace("\n","")
                    return domain_name

    def _generate_objects_line(self):
        objects = self.detector.get_objects()
        # generates a string for the line needed in problem file
        objects_line = ""
        for type in objects:
            objects_line += " ".join(objects[type])
            objects_line += " - " + type + " "
        return objects_line

    def _generate_init_str(self):
        # generate string with initial state
        init = self.detector.detect()
        init_str = " ".join(init)
        return init_str

    def act(self, op_name, fluent_state=True) -> List[str]:
        state, reward, done, info = self.executor.act(op_name)
        if fluent_state:
            state = self.detector.interpret(state)
        return state, reward, done, info

    def perceive(self) -> List[str]:
        """
        Returns fluent state
        """
        return self.detector.detect()

    def check_if_formula_holds(self, formula):
        """
        Returns true if formula (conjunction) holds in current state
        """
        fluent_state = self.detector.detect()
        if set(formula).issubset(set(fluent_state)):
            return True
        else:
            return False

    @env.setter
    def env(self, env):
        self._env = env


class Performance:
    """
    Class instantiated in Brain for maintaining various performance-related metrics
    """

    def __init__(self):
        self._no_of_steps = 0
        self._goal_achieved = False
        self._needed_exploration = False
        self._cumulative_reward = 0.
        self._operators_discovered = 0.

    @property
    def operators_discovered(self) -> int:
        return self._operators_discovered

    def discovered_operator(self):
        self._operators_discovered += 1

    @property
    def cumulative_reward(self) -> float:
        return self._cumulative_reward

    @cumulative_reward.setter
    def cumulative_reward(self, value):
        self._cumulative_reward = value

    @property
    def needed_exploration(self) -> bool:
        return self._needed_exploration

    @property
    def no_of_steps(self) -> int:
        return self._no_of_steps

    @needed_exploration.setter
    def needed_exploration(self, value):
        self._needed_exploration = value

    @property
    def goal_achieved(self):
        return self._goal_achieved

    @goal_achieved.setter
    def goal_achieved(self, value):
        self._goal_achieved = value

    def increment_step(self, n: int = 1):
        """
        Increment steps by n
        :param n: int
        """
        self._no_of_steps += n


class LTM:
    """
    Class instantiated in Brain for long term memory
    """

    def __init__(self, history=None):
        if history is None:
            history = []
        self._history = history
        self._cached_plans = dict()
        self._unplannable_states = {}

    @property
    def history(self) -> List[Tuple[Operator, List[Predicate]]]:
        return self._history

    @history.setter
    def history(self, history):
        self._history = history

    def add_to_history(self, entry: Tuple[Operator, List[Predicate]]):
        """
        Append an entry to the history at the end
        :param entry:
        :return:
        """
        hist = self.history
        hist.append(entry)
        self.history = hist

    def recall_last(self) -> Tuple[Operator, List[Predicate]]:
        """
        Returns last action and consequent state
        :return:
        """
        return self._history[len(self.history) - 1]

    @property
    def cached_plans(self):
        return self._cached_plans

    @property
    def unplannable_states(self):
        return self._unplannable_states

    @unplannable_states.setter
    def unplannable_states(self, value):
        self._unplannable_states = value


class WM:
    """
    Class instantiated in Brain for Working Memory
    """

    def __init__(self, domain_bias, problem_bias, operators=None, freeze_task=False):
        self._problem = self._parse(domain_bias, problem_bias)
        self._task = self._ground(self._problem)
        if operators is not None:
            for operator in operators:
                self._task.add_operator(operator)
        self._current_plan = []
        self._operator_stack = [] # Important. The agent pulls from this stack to execute actions.
        self._current_operator = None
        self._current_op_idx = -1
        self._situation = {}
        self._forward_states = {}
        self._backward_conditions = {}
        self._found_operators = set()
        self._freeze_task = freeze_task

    def found_operators(self):
        return frozenset(self._found_operators)

    def add_operator(self, operator):
        if not self._freeze_task:
            self._task.add_operator(operator)
        self._found_operators.add(operator)

    def delete_operator(self, operator):
        if not self._freeze_task:
            self._task.operators.remove(operator)
            self._found_operators.remove(operator)

    @property
    def forward_states(self):
        return self._forward_states

    @forward_states.setter
    def forward_states(self, forward_states):
        self._forward_states = forward_states

    @property
    def backward_conditions(self):
        return self._backward_conditions

    @backward_conditions.setter
    def backward_conditions(self, backward_conditions):
        self._backward_conditions = backward_conditions

    @property
    def task(self) -> Task:
        return self._task

    @task.setter
    def task(self, task: Task):
        self._task = task

    @property
    def situation(self) -> Dict:
        return self._situation

    @situation.setter
    def situation(self, situation: Dict):
        self._situation = situation

    @property
    def current_plan(self) -> List[Operator]:
        return self._current_plan

    @current_plan.setter
    def current_plan(self, current_plan: List[Operator]):
        self._current_plan = current_plan
        self.add_to_operator_stack(self.current_plan) # everytime you set plan, you also append it to the stack

    @property
    def operator_stack(self) -> List[Operator]:
        return self._operator_stack

    def add_to_operator_stack(self, ops):
        self.operator_stack.extend(ops)

    def clear_operator_stack(self):
        self._operator_stack = []

    @property
    def current_operator(self) -> Operator:
        return self._current_operator

    @current_operator.setter
    def current_operator(self, op: Operator):
        self._current_operator = op

    def get_idx_current_operator(self) -> int:
        """
        Get index of current operator in plan
        :return:
        """
        plan = self.current_plan
        operator = self.current_operator
        if plan:
            for idx, op in enumerate(plan):
                if op.name == operator.name:
                    return idx
            log.error("Current operator not in plan")
            return -1
        else:
            log.error("Plan is empty")
            return -1

    def get_next_operator(self) -> Tuple[Operator, int]:
        """
        Pops the operator from the stack.
        :return:
        """
        if self.operator_stack:
            op = self.operator_stack.pop(0)
            self.current_operator = op
            idx = self.get_idx_current_operator()
            return op, idx
        else:
            return None

    def see_next_operator(self) -> Operator:
        if self.operator_stack:
            return self.operator_stack[0]
        else:
            log.info("No more operators in the stack")
            return None

    def _parse(self, domain_file, problem_file) -> Problem:
        """
        Get problem from domain and problem pddl files
        :param domain_file: pddl domain file
        :param problem_file: pddl problem file
        :return: Problem instance
        """
        parser = Parser(domain_file, problem_file)
        domain = parser.parse_domain()
        problem = parser.parse_problem(domain)
        return problem

    def _ground(self, problem) -> Task:
        """
        Grounds problem into STRIPS task
        :param problem: Problem instance
        :return: Task instance
        """
        task = grounding.ground(problem)
        return task

class Learning:
    def __init__(self):
        self._learners = []
        self._found_list = []

    @property
    def learners(self):
        return self._learners

    @learners.setter
    def learners(self, learner):
        self._learners = learner

    @property
    def found_list(self):
        return self._found_list

    def clear_found_list(self):
        self._found_list.clear()

    def add_to_found_list(self, operator):
        self._found_list.append(operator)


class Brain:
    """
    Contains all the stuff the agent needs to maintain and update
    STRIPS task info, STRIPS problem info, history, oracle etc.
    """

    def __init__(self, env, domain_bias, goal, detector, executor_class, state_hasher=None, executor=None,
                 operators=None, operator_filename=None, render=None, freeze_task=False, ops_every=1):
        self.configuration = Configuration()
        self.affect = Affect()
        self.performance = Performance()
        self.motor = Motor(env=env, domain_bias=domain_bias, detector=detector, executor_class=executor_class,
                           state_hasher=state_hasher, executor=executor, render=render)
        problem = self.motor.get_initial_problem(goal) # looks at env and constructs an initial problem file
        self.wm = WM(domain_bias=domain_bias, problem_bias=problem, operators=operators, freeze_task=freeze_task)
        self.ltm = LTM(history=[(None, self.wm.task.initial_state)])
        self.learning = Learning()
        self.operator_filename = operator_filename
        self.episode_counter = 0
        self.ops_every = ops_every


if __name__ == "__main__":
    from agent.detection.cupsworld_detector import CupsWorldDetector
    from agent.execution.cupsworld_executor import CupsWorldExecutor

    env = gym.make('CupsWorld-v0')
    domain = "../domains/domain.pddl"
    goal = "(and (inside block_blue cup) (upright cup))"
    brain = Brain(env=env, domain_bias=domain, goal=goal, detector=CupsWorldDetector, executor=CupsWorldExecutor)
