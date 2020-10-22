import itertools
import logging
import csv

from agent.checking_mode import CheckingMode
from agent.execution_mode import ExecutionMode
from agent.explorer.policy_explorer import PolicyExplorer
from agent.learning.operator_learner import TabularOperatorUnionLearner
from agent.mode import Mode
from representation.task import Operator

from helper import get_root_dir
logging.basicConfig(filename=get_root_dir() + "/logs/console.log",
                    format="%(name)s — %(levelname)s — %(message)s",
                    filemode='w')
log = logging.getLogger("exploration_mode")
log.setLevel(logging.DEBUG)


class ExplorationMode(Mode):
    def __init__(self, brain, explorer_class=PolicyExplorer):
        self.name = "Creative Exploration Mode"
        super().__init__(brain)
        self.done_exploring = False
        self.explorer_class = explorer_class

    def _run(self):
        """
        (1) get models
        (2) Initialize explorers
        (3) If learners not already initialized, initialize them
        (4) found_list = []
        (5) if found_list:
                self.done_exploring = True
            else:
                explorer.explore()
        """

        self.brain.performance.needed_exploration = True
        if not self.brain.ltm.unplannable_states:
            self.brain.ltm.unplannable_states = set(self.brain.wm.forward_states)

        # Initialize online explorer/ experimenter
        log.info("Initializing Explorer")

        if not self.brain.learning.learners:
            # initialize learners
            online_learner = TabularOperatorUnionLearner(self.brain.motor.env, self.brain.motor.state_hasher.hash)
            self.brain.learning.learners = [online_learner]
            self.brain.motor.executor.attach_learner(online_learner)
            log.info("Online learner initialized for the first time")

        learners = self.brain.learning.learners

        explorer = self.explorer_class(self.brain)

        log.info("Begin exploration...")

        explorer.explore()

        if self.brain.episode_counter % self.brain.ops_every == 0: # TODO definitely change this to a global setting.
            operators = list(itertools.chain(*[learner.get_operator_confidences(0.9, self.brain.wm.forward_states)
                                               for learner in learners[1:]]))

            operators = filter(lambda x: x[2] > 0.9, operators)

            # operators = filter(lambda x: any(x[1].applicable(s) for s in self.brain.wm.forward_states), operators)

            operators = list(operators)

            all_ops_to_remove = set()
            old_ops = self.brain.wm.found_operators()
            for learner, operator, confidence in operators:
                found_better = False
                ops_to_remove = []
                for op in self.brain.wm.found_operators(): # For now, we're not going to replace preexisting operators
                    if self.operator_better(op, operator):
                        found_better = True
                        break
                    elif self.operator_better(operator, op) and not self.equal_ignore_name(operator, op):
                        # Replace operators that are strictly worse than the operator we found
                        ops_to_remove.append(op)
                all_ops_to_remove |= set(ops_to_remove)
                if not found_better:
                    log.info("Operator found!")
                    log.info(operator.__str__())
                    self.brain.performance.discovered_operator()
                    action_name = self.brain.motor.executor.add_action(learner, [], learner.execute,
                                                                       action_name=None)
                    revised_op = self._remove_params_and_rename(operator, action_name)
                    self.brain.wm.add_operator(revised_op)
                    self.output_operator(revised_op)
                    self.output_abbreviated_op(revised_op, self.brain.learning.learners.index(learner))
                    for op in ops_to_remove:
                        self.output_op_edge(op, revised_op)
                    self.brain.ltm.unplannable_states = set()  # We have new operators; we definitely need to replan.
            for op in all_ops_to_remove:
                self.brain.wm.delete_operator(op)
                self.output_deleted_op(op)

        if self.brain.wm.current_plan:
            log.info("We can plan from here! Switching to execution mode.")
            log.info("Plan: " + str(self.brain.wm.current_plan))
        else:
            log.info("Episode terminated without finding a plan.")

        self.done_exploring = True

    def output_abbreviated_op(self, operator, learner_index):
        if self.brain.operator_filename:
            with open(self.brain.operator_filename + "_abbrev", 'a') as f:
                writer = csv.writer(f)
                writer.writerow([self.brain.episode_counter, operator.name,
                                 len(operator.pos_preconditions) + len(operator.neg_preconditions), learner_index])

    def output_op_edge(self, worse_op, better_op):
        if self.brain.operator_filename:
            with open(self.brain.operator_filename + "_edge", 'a') as f:
                writer = csv.writer(f)
                writer.writerow([worse_op.name, better_op.name])

    def output_deleted_op(self, operator):
        if self.brain.operator_filename:
            with open(self.brain.operator_filename, 'a') as f:
                f.write(str(self.brain.episode_counter))
                f.write(": REPLACED OPERATOR ")
                f.write(operator.name)
                f.write("\n")
            f.close()

    def output_operator(self, operator):
        if self.brain.operator_filename:
            with open(self.brain.operator_filename, 'a') as f:
                f.write(str(self.brain.episode_counter))
                f.write("\n")
                f.write(str(operator))
                f.write("-------------------")
            f.close()

    def operator_better(self, op1, op2):
        # op1 is better than op2 if it applies more generally and has the same effect, but is more specific.
        # For effects, op1 should at least have all the add, delete, and static effects of op2.
        # We allow it to have other add, delete, and static effects only if those are unknown effects in op2.
        # We're using this algorithm to ensure that the operator created does something which existing operators don't
        # already do.
        op1_statics = self.brain.wm.task.facts - op1.statics if op1.invert_statics else op1.statics
        op2_statics = self.brain.wm.task.facts - op2.statics if op2.invert_statics else op2.statics
        return (op1.pos_preconditions <= op2.pos_preconditions \
                and op1.neg_preconditions <= op2.neg_preconditions
                and op1.add_effects >= op2.add_effects and op1.del_effects >= op2.del_effects
                and op1_statics >= op2_statics
                and len(((op1.add_effects - op2.add_effects) | (op1.del_effects - op2.del_effects) |
                         (op1_statics - op2_statics)) & (op2.add_effects | op2.del_effects | op2_statics)) == 0)

    def equal_ignore_name(self, op1, op2):
        return (
                op1.invert_statics == op2.invert_statics
                and op1.statics == op2.statics
                and op1.pos_preconditions == op2.pos_preconditions
                and op1.neg_preconditions == op2.neg_preconditions
                and op1.add_effects == op2.add_effects
                and op1.del_effects == op2.del_effects
        )

    def _remove_params_and_rename(self, op, name):
        return Operator(name, op.pos_preconditions, op.neg_preconditions, op.invert_statics, op.statics, op.add_effects,
                        op.del_effects)

    def _next(self):
        if self.brain.wm.current_plan:
            return CheckingMode(self.brain)
        else:
            # Environment was reset; we'll need to plan from the beginning again.
            return None
