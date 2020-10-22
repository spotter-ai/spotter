#
# This file is part of pyperplan.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>
#

"""
Classes for representing an open-world planning task
"""
from functools import reduce


class Operator:
    """
    The preconditions represent the facts that have to be true
    before the operator can be applied.
    statics are the fluents whose values we know will not change when a fluent is done.
    add_effects are the facts that the operator makes true.
    delete_effects are the facts that the operator makes false.
    Any fluent in the state that is neither in the statics, the add effects, or the del effects, will be unknown
    when the operator is complete.
    """

    def __init__(self, name, pos_preconds, neg_preconds, default_static, statics, add_effects, del_effects):
        self.name = name
        self.invert_statics = default_static
        self.statics = frozenset(statics)
        self.pos_preconditions = frozenset(pos_preconds)
        self.neg_preconditions = frozenset(neg_preconds)
        self.add_effects = frozenset(add_effects)
        self.del_effects = frozenset(del_effects)

    def applicable(self, state):
        """
        A fluent state in this form takes the form of a dict with fluent names mapping to true or false, and where
        unknown fluents are absent.
        Operators are applicable when their set of preconditions is a subset
        of the facts that are true in "state".
        @return True if the operator's preconditions is a subset of the state with matching truth values
                False otherwise
        """
        if not isinstance(state, tuple):
            # The detector returns states that don't specify both positive and negative preconditions; we assume that
            # everything that's not True in it is False.
            return self.pos_preconditions <= state and len(self.neg_preconditions & state) == 0
        return self.pos_preconditions <= state[0] and self.neg_preconditions <= state[1]

    def relevant(self, goal):
        """
        preconditions and goals can include positive and negative fluents.

        Operators are relevant when all the following conditions hold, where the goal has x_i := c_i for c_i in {T, F}:
        (1a) for all x_i = c_i in g, if c_i=T then x_i in add or static fluents;
        (1b) if c_i=F then x_i in del effects or static fluents;
        (2) at least one x_i must be in the add or delete effects of the operator.
        (3) if x_i = c_i in g, x_i = c_i' in pre(operator), and x_i in statics(operator) then c_i = c_i'
        """
        cond2 = False
        pos_goal_conditions = goal[0]
        neg_goal_conditions = goal[1]
        for x in pos_goal_conditions:
            if (self.invert_statics ^ (x not in self.statics)) and x not in self.add_effects:
                return False
            elif x in self.del_effects:
                return False
            if x in self.add_effects:
                cond2 = True
            elif (self.invert_statics ^ (x in self.statics)) and x in self.neg_preconditions:
                # elif because if default_static is true then a variable can be "static" in that it's not in
                # self.statics, but you still need to make sure it's not in the add effects
                return False
        for x in neg_goal_conditions:
            if (self.invert_statics ^ (x not in self.statics)) and x not in self.del_effects:
                return False
            elif x in self.add_effects:
                return False
            if x in self.del_effects:
                cond2 = True
            elif (self.invert_statics ^ (x in self.statics)) and x in self.pos_preconditions:
                return False
        return cond2

    def regress(self, goal):
        """
        Regressing an operator returns an initial_condition which contains
        (1) the preconditions of the operator
        (2) any static fluents that need to remain constant in g
        """
        # assert self.relevant(goal)
        if self.invert_statics:
            return self.pos_preconditions | (goal[0] - self.add_effects), \
                   self.neg_preconditions | (goal[1] - self.del_effects)
        else:
            return self.pos_preconditions | ((goal[0] - self.add_effects) & self.statics), \
                   self.neg_preconditions | ((goal[1] - self.del_effects) & self.statics)

    def apply(self, state):
        """
        Applying an operator means removing the facts that are made false
        by the operator from the set of true facts in state and adding
        the facts made true.  Facts in statics retain their truth values, and all other facts
        become unknown.

        Note that therefore it is possible to have operands that make a
        fact both false and true. This results in the fact being true
        at the end.
        @param state The state that the operator should be applied to
        @return A new state (set of facts) after the application of the
                operator
        """
        assert self.applicable(state)
        if self.invert_statics:
            pos_fluents = (state[0] - self.del_effects - self.statics) | self.add_effects
            neg_fluents = (state[1] - self.add_effects - self.statics) | self.del_effects
        else:
            # This should work because the statics are guaranteed not to have predicates that are in add_effects or
            # del_effects.
            pos_fluents = (self.statics & state[0] - self.del_effects) | self.add_effects
            neg_fluents = (self.statics & state[1] - self.add_effects) | self.del_effects
        # Fluents that end up both positive and negative become unknown.
        return pos_fluents - neg_fluents, neg_fluents - pos_fluents

    def __eq__(self, other):
        return (
            self.name == other.name
            and self.invert_statics == other.invert_statics
            and self.statics == other.statics
            and self.pos_preconditions == other.pos_preconditions
            and self.neg_preconditions == other.neg_preconditions
            and self.add_effects == other.add_effects
            and self.del_effects == other.del_effects
        )

    def __hash__(self):
        return hash((self.name, self.statics, self.pos_preconditions, self.neg_preconditions, self.invert_statics,
                     self.add_effects, self.del_effects))

    def __str__(self):
        s = "%s\n" % self.name
        for group, facts in [
            ("+PR", self.pos_preconditions),
            ("-PR", self.neg_preconditions),
            ("UNK" if self.invert_statics else "STA", self.statics),
            ("ADD", self.add_effects),
            ("DEL", self.del_effects),
        ]:
            for fact in facts:
                s += "  {}: {}\n".format(group, fact)
        return s

    def __repr__(self):
        return "<Op %s>" % self.name


class Task:
    """
    A STRIPS planning task
    """

    def __init__(self, name, facts, initial_state, goals, operators, statics=None):
        """
        @param name The task's name
        @param facts A set of all the fact names that are valid in the domain
        @param initial_state A set of fact names that are true at the beginning
        @param goals A dictionary of fluents and the values they must take to solve the problem
        @param operators A set of open world operator instances for the domain
        """
        self.name = name
        self.facts = facts
        self.initial_state = (frozenset(initial_state), frozenset(facts - initial_state))
        self.goals = goals
        self.operators = operators
        self.statics = statics

    def goal_reached(self, state):
        """
        The goal has been reached if all facts that are true in "goals"
        are true in "state".
        @return True if all the goals are reached, False otherwise
        """
        if not isinstance(state, tuple):
            # State is just the positive fluents; treat it accordingly.
            return self.goals[0] <= state and len(self.goals[1] & state) == 0
        return self.goals[0] <= state[0] and self.goals[1] <= state[1]

    def start_reached(self, condition):
        """
        A start stated has been reached if the initial state satisfies condition
        """
        return condition[0] <= self.initial_state[0] and condition[1] <= self.initial_state[1]

    def get_successor_states(self, state):
        """
        @return A list with (op, new_state) pairs where "op" is the applicable
        operator and "new_state" the state that results when "op" is applied
        in state "state".
        """
        return [(op, op.apply(state)) for op in self.operators if op.applicable(state)]

    def get_regressor_conditions(self, condition):
        """
        Given a condition, returns a set of conditions
        """
        return [(op, op.regress(condition)) for op in self.operators if op.relevant(condition)]
        # out = []
        # for op in self.operators:
        #     if op.relevant(condition):
        #         out.append(op.regress(condition))
        # return out

    def add_operator(self, operator):
        self.operators.append(operator)

    def replace_operator(self, operator):
        for idx, op in enumerate(self.operators):
            if op.name == operator.name:
                self.operators[idx] = operator

    def copy_with_new_initial_state(self, init):
        return Task(self.name, self.facts, init, self.goals, self.operators, self.statics)

    def __str__(self):
        s = "Task {0}\n  Vars:  {1}\n  Init:  {2}\n  Goals: {3}\n  Ops:   {4}"
        return s.format(
            self.name,
            ", ".join(self.facts),
            self.initial_state,
            self.goals,
            "\n".join(map(repr, self.operators)),
        )

    def __repr__(self):
        string = "<Task {0}, vars: {1}, operators: {2}>"
        return string.format(self.name, len(self.facts), len(self.operators))


def state_subsumes(fs1, fs2):
    return fs1[0] >= fs2[0] and fs1[1] >= fs2[1]