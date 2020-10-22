from agent.explorer.explorer import Explorer
from agent.planning.forward_search import ForwardSearch
from agent.learning.operator_learner import TabularOperatorLearner
from agent.learning.policy import EpsilonGreedyPolicy
from representation.task import state_subsumes, Operator


class PolicyExplorer(Explorer):
    def __init__(self, brain):
        self.brain = brain
        self.env = self.brain.motor.env
        self.learners = self.brain.learning.learners
        self.policy = EpsilonGreedyPolicy(self.env, self.learners[0], self.brain.motor.epsilon)
        # self.policy = RandomDiscretePolicy(self.env)
        self.detector = self.brain.motor.detector
        self.executor = self.brain.motor.executor
        self.operator_idx = 0

    def _get_operator(self, fs):
        self.operator_idx += 1
        return Operator("op" + str(self.operator_idx - 1).zfill(5), set(), set(), False, set(), fs[0], fs[1])

    def _explore(self):
        done = False
        hit_backward_state = False

        all_states = []
        s = self.env.last_observation()

        # This needs to be stored globally.
        while not hit_backward_state:
            a = self.policy.action(s)
            all_states.append((s, a))
            sp, reward, done, info = self.executor.execute_core_action(a)
            self.brain.performance.cumulative_reward += reward

            pos_state = frozenset(self.detector.interpret(sp))
            fs = pos_state, frozenset(self.brain.wm.task.facts - pos_state)

            for ps in self.brain.ltm.cached_plans.keys():
                if state_subsumes(fs, ps):
                    self.brain.wm.current_plan = self.brain.ltm.cached_plans[ps]
                    hit_backward_state = True
                    break
            if not hit_backward_state and fs not in self.brain.ltm.unplannable_states:
                search = ForwardSearch()
                task = self.brain.wm.task.copy_with_new_initial_state(pos_state)
                plan, frontier, visited = search.search(task, set(self.brain.ltm.unplannable_states))
                if plan:
                    generalized_fs = task.goals
                    for i in range(1, len(plan) + 1):
                        generalized_fs = plan[-i].regress(generalized_fs)
                        if generalized_fs not in self.brain.ltm.cached_plans:
                            self.brain.ltm.cached_plans[generalized_fs] = plan[-i:]
                            learner = TabularOperatorLearner(self.env, self.brain.motor.state_hasher.hash, self.executor,
                                                             self._get_operator(generalized_fs),
                                                             self.detector)
                            self.learners.append(learner)
                            # New learners need to learn from that first sweet timestep; otherwise they'll learn from the
                            # executor.
                            learner.train(s, a, reward, sp)
                            self.learners[0].add_learner_and_revise_reward(learner, s, a, reward, sp)
                            self.executor.attach_learner(learner)
                    hit_backward_state = True
                    self.brain.wm.current_plan = self.brain.ltm.cached_plans[generalized_fs]
                else:
                    self.brain.ltm.unplannable_states = visited

            s = sp
            if done:
                self.brain.wm.current_plan = []
                # self.env.reset()
                break
