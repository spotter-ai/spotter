import heapq
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from random import random

import numpy as np

from agent.execution.executor import RewardAccumulator, EpisodeTerminatedException
from agent.learning.learner import Learner, CustomRewardTabularLearner
from representation.task import Operator, state_subsumes
from functools import partial

import logging
from helper import get_root_dir
logging.basicConfig(filename=get_root_dir() + "/logs/console.log",
                    format="%(name)s — %(levelname)s — %(message)s",
                    filemode='w')

# Creating an object
log = logging.getLogger("operator_learner")

# Setting the threshold of logger to DEBUG
log.setLevel(logging.DEBUG)

class OperatorLearner(Learner, ABC):
    def execute(self):
        return self._execute()

    @abstractmethod
    def _execute(self):
        pass

    def get_operator_confidences(self, threshold, applicable_states=None):
        return self._get_operator_confidences(threshold, applicable_states)

    @abstractmethod
    def _get_operator_confidences(self, threshold, applicable_states):
        pass


class OperatorSparseRewardFunction:
    """
    A class that provides a sparse reward function corresponding to a given planning operator.
    """

    def __init__(self, operator, detector):
        self.operator = operator
        self.detector = detector

    def reward(self, s, a, r, sp):
        fs = set(self.detector.interpret(sp))
        if self.operator.add_effects <= fs and len(self.operator.del_effects & fs) == 0:
            return 1.
        else:
            return 0.


class TabularOperatorLearner(CustomRewardTabularLearner, OperatorLearner):

    def _execute(self):
        # Execute the learned policy until the goal is reached, or until a timeout.
        accumulator = RewardAccumulator(self.env, action_timeout=100, discount=self.discount)
        obs = self.env.last_observation()
        try:
            while not satisfies_operator(obs, self.operator, self.detector):
                action = self.action(obs)
                obs, reward, done, info = self.executor.execute_core_action(action, accumulator)
        except EpisodeTerminatedException:
            pass
        finally:
            return accumulator.combine_steps()

    def _get_fluents(self, s):
        pos_state = frozenset(self.detector.interpret(s))
        return pos_state, frozenset(set(self.detector.all_fluents()) - pos_state)

    # These all assume that fs1 is the more general operator, and fs2 is a specific operator with which you want to
    # build conflicts.
    def _knowledge_conflicts(self, fs1, fs2):
        return len(fs1[1] & fs2[0]) + len(fs1[0] & fs2[1]) > 0

    def _pos_conflict_set(self, fs1, fs2):
        assert not self._knowledge_conflicts(fs1, fs2)
        return fs2[1] - fs1[1]

    def _neg_conflict_set(self, fs1, fs2):
        return fs2[0] - fs1[0]

    def _specify(self, fs, unique_fs):
        common_knowledge = None
        for pos, neg in unique_fs:
            if not state_subsumes((pos, neg), fs):
                continue
            if common_knowledge is None:
                common_knowledge = pos, neg
            else:
                common_knowledge = common_knowledge[0] & pos, common_knowledge[1] & neg
        return common_knowledge

    def _expand_specific_node(self, V, specific_frontier, value_threshold, specific_fs, specific_fs_values, num_states,
                              above_threshold):
        num_preconds, (pos, neg) = heapq.heappop(specific_frontier)
        for fs in reversed(specific_fs):
            common_knowledge = (frozenset(fs[0] & pos), frozenset(fs[1] & neg))
            if common_knowledge not in self._precondition_values:
                value = self.get_avg_for_precondition(specific_fs_values, num_states, common_knowledge)
                self._precondition_values[common_knowledge] = value
                self._maybe_add_specific(common_knowledge, value, value_threshold, specific_frontier, above_threshold)

    def _maybe_add_specific(self, preconds, value, value_threshold, specific_frontier, above_threshold):
        if value > value_threshold:
            if not any(state_subsumes(preconds, fs) for fs, v in above_threshold):
                for fs, v in set(above_threshold):
                    if state_subsumes(fs, preconds):
                        above_threshold.remove((fs, v))
                above_threshold.add((preconds, value))
            # specific_frontier.insert(0, preconds)
            heapq.heappush(specific_frontier, (len(preconds[0]) + len(preconds[1]),
                                               preconds))


    def _maybe_add_general(self, preconds, value, value_threshold, general_frontier, above_threshold):
        if value > value_threshold:
            for fs, v in set(above_threshold):
                if state_subsumes(fs, preconds):
                    above_threshold.remove((fs, v))
            above_threshold.add((preconds, value))
        else:
            general_frontier.insert(0, preconds)

    def _expand_general_node(self, V, general_frontier, applicable_states, value_threshold, specific_fs,
                             specific_fs_values, num_states, all_fluents, above_threshold):
        pos, neg = general_frontier.pop()
        unused_fluents = set(all_fluents) - pos - neg
        new_pos_conditions = set(self._specify((frozenset(pos | {pos_add}), neg), specific_fs)
                                 for pos_add in unused_fluents)
        new_neg_conditions = set(self._specify((pos, frozenset(neg | {neg_add})), specific_fs)
                                 for neg_add in unused_fluents)
        new_conditions = new_pos_conditions | new_neg_conditions
        for new_condition in new_conditions:
            # Exclude anything that doesn't apply in at least one of the applicable states
            if new_condition not in self._precondition_values.keys() and \
               any(state_subsumes(app_state, new_condition) for app_state in applicable_states):
                value = self.get_avg_for_precondition(specific_fs_values, num_states, new_condition)
                self._precondition_values[new_condition] = value
                self._maybe_add_general(new_condition, value, value_threshold, general_frontier, above_threshold)


    def get_avg_for_precondition(self, unique_fs_values, num_states, precond):
        relevant_fs = [specific_fs for specific_fs in unique_fs_values
                       if state_subsumes(specific_fs, precond)]
        if len(relevant_fs) == 0:
            return 0.
        num = 0.
        denom = 0.
        for fs in relevant_fs:
            num += unique_fs_values[fs]*num_states[fs]
            denom += num_states[fs]
        return num/denom

    def _compute_precondition_values(self, value_threshold, applicable_states=None, num_operators=1E6,
                                     max_conditions=1E6, prob_general=0.):

        start_time = time.time()
        all_fluents = self.detector.all_fluents()
        hash_to_fluents = self._hash_to_fluents
        V = {hs: np.max(self.Q[hs]) for hs in hash_to_fluents.keys()}
        above_threshold = set()

        # This is linear time in total states, rather than quadratic.
        unique_fs_values = defaultdict(float)
        num_states = defaultdict(int)
        for hs in V:
            fs = hash_to_fluents[hs]
            unique_fs_values[fs] += V[hs]
            num_states[fs] += 1

        unique_fluent_states = list(set(unique_fs_values.keys()))
        if applicable_states is None:
            applicable_states = set(unique_fluent_states)

        for fs in unique_fluent_states:
            unique_fs_values[fs] /= float(num_states[fs])

        unique_fluent_states.sort(key=lambda fs: unique_fs_values[fs])

        specific_frontier = []

        if len(self._precondition_values) > 0:
            for fs in self._precondition_values.keys():
                old_v = self._precondition_values[fs]
                precond_value = self.get_avg_for_precondition(unique_fs_values, num_states, fs)
                self._precondition_values[fs] = precond_value
                list_to_add = specific_frontier if old_v <= value_threshold else []
                self._maybe_add_specific(fs, precond_value, value_threshold, list_to_add, above_threshold)
        else:
            # Initialize the specific-to-general search by spawning with all applicable states.
            for fs in applicable_states:
                precond_value = self.get_avg_for_precondition(unique_fs_values, num_states, fs)
                self._precondition_values[fs] = precond_value
                self._maybe_add_specific(fs, precond_value, value_threshold, specific_frontier, above_threshold)

        # Initialize the general-to-specific search by spawning with the common knowledge among all states visited.
        most_general_condition = self._specify((frozenset(), frozenset()), unique_fluent_states)
        general_frontier = []
        value = self.get_avg_for_precondition(unique_fs_values, num_states, most_general_condition)
        self._precondition_values[most_general_condition] = value
        self._maybe_add_general(most_general_condition, value, value_threshold, general_frontier, above_threshold)

        while len(general_frontier) > 0 and len(specific_frontier) > 0 \
                and len(self._precondition_values) <= max_conditions \
                and len(above_threshold) < num_operators:
            expand_general_node = random() < prob_general
            if expand_general_node:
                self._expand_general_node(V, general_frontier, applicable_states, value_threshold, unique_fluent_states,
                                          unique_fs_values, num_states, all_fluents, above_threshold)
            else:
                self._expand_specific_node(V, specific_frontier, value_threshold, unique_fluent_states,
                                           unique_fs_values, num_states, above_threshold)
        log.debug("Elapsed time: {}".format(time.time() - start_time))
        log.debug("Total # preconditions checked: {}".format(len(self._precondition_values)))
        log.debug("# preconditions above threshold: {}".format(len(above_threshold)))
        return above_threshold

    def _get_operator_confidences(self, threshold, applicable_states=None):
        best_preconds = self._compute_precondition_values(threshold, applicable_states)
        return [(self, self._op_with_new_preconditions(self.operator, precond), v) for precond, v in best_preconds]

    def _op_with_new_preconditions(self, op, new_preconditions):
        return Operator(op.name, new_preconditions[0], new_preconditions[1], op.invert_statics, op.statics,
                        op.add_effects, op.del_effects)

    def __init__(self, env, hash_function, executor, operator, detector, learning_rate=0.1, discount=0.99,
                 reward_class=OperatorSparseRewardFunction):
        super().__init__(env, hash_function, reward_class(operator, detector), learning_rate, discount)
        self.operator = operator
        self.executor = executor  # We need this primarily so that we can execute the policy once we're done.
        self.accumulator = self.env # Also for execution purposes. TODO at some point find a cleaner way to do this.
        self.detector = detector
        self._hash_to_fluents = {}
        self._precondition_values = {}

    def _train(self, s, a, r, sp):
        hs = self._hash_function(s)
        hsp = self._hash_function(sp)
        pos, neg = self._get_fluents(s)
        if not fs_satisfies_operator(pos, self.operator):
            super()._train_hashed(hs, a, self.reward_fn.reward(s, a, r, sp), hsp)
        if hs not in self._hash_to_fluents:
            self._hash_to_fluents[hs] = (pos, neg)
        # if hsp not in self._hash_to_fluents:
        #     self._hash_to_fluents[hsp] = self._get_fluents(sp)


def satisfies_operator(obs, operator, detector):
    fs = set(detector.interpret(obs))
    return operator.add_effects <= fs and len(operator.del_effects & fs) == 0


def fs_satisfies_operator(fs, operator):
    return operator.add_effects <= fs and len(operator.del_effects & fs) == 0


class TabularOperatorUnionLearner(CustomRewardTabularLearner):
    def __init__(self, env, hash_function, learners=None, learning_rate=0.1, discount=0.99):
        if learners is None:
            learners = []
        self.learners = list(learners)
        super().__init__(env, hash_function, self, learning_rate, discount)

    def reward(self, s, a, r, sp):
        if len(self.learners) > 0:
            detector = self.learners[0].detector
            fs = set(detector.interpret(sp))
            if any(fs_satisfies_operator(fs, learner.operator) for learner in self.learners):
                return 1.
        return 0.

    def add_learner_and_revise_reward(self, learner, s, a, r, sp):
        # We added a learner after we already learned the reward, so we need to modify the Q-function to reflect the new
        # reward.
        # assumes that s is not equal to sp, or else this doesn't quite accurately update the Q-function. Should be fine
        # for our purposes, though.
        orig_reward = self.reward(s, a, r, sp)
        self.learners.append(learner)
        new_reward = self.reward(s, a, r, sp)
        self.Q[self._hash_function(s)][a] += self.learning_rate*(new_reward - orig_reward)

    def _train(self, s, a, r, sp):
        if len(self.learners) > 0:
            detector = self.learners[0].detector
            fs = set(detector.interpret(s))
            if not any(fs_satisfies_operator(fs, learner.operator) for learner in self.learners):
                super()._train(s, a, r, sp)

