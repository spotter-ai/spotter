from itertools import product

from agent.env_wrappers import LastObsWrapper
from agent.execution.executor import Executor, operator_parts


class ExtendableExecutor(Executor):

    def _remove_learner(self, learner):
        self._base_executor.remove_learner(learner)

    def _execute_core_action(self, action, accumulator):
        return self._base_executor.execute_core_action(action, accumulator)

    def _attach_learner(self, learner):
        self._base_executor.attach_learner(learner)

    def _clear_learners(self):
        self._base_executor.clear_learners()

    def set_environment(self, env):
        self.env = env
        self._base_executor.env = env

    def set_record_path(self, episode_number):
        self._base_executor.set_record_path(episode_number)

    # TODO we need to be able to produce an executor that can be extended with as many actions as we might want.
    # To do this we need to make sure that we can execute particular methods from particular other executors. A few
    # potential ways to do this:
    # (a) force all executor methods to be static (have to pass in the environment, detectors, etc)
    # (b) store the instance associated with each method; that way, the method can always be called on the instance
    def __init__(self, env, detector=None, base_executor=None):
        super().__init__()
        self._unnamed_executors_counter = 0
        self.env = env
        self._executor_map = {}
        self._detector = detector
        self._base_executor = base_executor
        if base_executor is not None:
            self._executor_map = dict(base_executor.executor_map())

    def executor_map(self):
        return self._executor_map

    def _all_actions(self):
        action_list = []
        type_to_objects = self._detector.get_objects(inherit=True)
        for executor_name in self._executor_map:
            possible_args = tuple([type_to_objects[t] for t in self._executor_map[executor_name][1]])
            for items in product(*possible_args):
                if len(items) == 0:
                    action_list.append("(" + executor_name + ")")
                else:
                    action_list.append("(" + executor_name + " " + " ".join(items) + ")")
        return action_list

    def rename_op(self, op, new_name):
        instance, signature, action_method = self._executor_map.pop(op.name)
        op.name = new_name
        self._executor_map[new_name] = (instance, signature, action_method)

    def _act(self, operator_name):
        executor_name, items = operator_parts(operator_name)
        instance, param_types, executor = self._executor_map[executor_name]
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

    def add_action(self, instance, signature, action_method, action_name=None):
        if action_name is None:
            action_name = 'new_action' + str(self._unnamed_executors_counter).zfill(4)
            self._unnamed_executors_counter += 1
        # Replaces the executor with the existing name with the new one if required.
        self._executor_map[action_name] = (instance, signature, action_method)
        return action_name