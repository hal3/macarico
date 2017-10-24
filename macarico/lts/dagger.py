from __future__ import division

import numpy as np
import macarico
from macarico.util import break_ties_by_policy

class DAgger(macarico.Learner):

    def __init__(self, reference, policy, p_rollin_ref, only_one_deviation=False):
        self.p_rollin_ref = p_rollin_ref
        self.policy = policy
        self.reference = reference
        self.objective = 0.0
        self.only_one_deviation = only_one_deviation
        self.t = None

#    @profile
    def __call__(self, state):
        if self.t is None:
            self.t = 0
            self.dev_t = np.random.choice(range(state.T)) + 1
        self.t += 1
        
        ref = break_ties_by_policy(self.reference, self.policy, state, False)
        pol = self.policy(state)
        if (not self.only_one_deviation) or (self.t == self.dev_t):
            self.objective += self.policy.forward(state, ref)

        return ref if self.p_rollin_ref() else pol

#    @profile
    def update(self, _):
        if not isinstance(self.objective, float):
            self.objective.backward()


class Coaching(DAgger):
    def __init__(self, reference, policy, p_rollin_ref, policy_coeff=0.):
        self.policy_coeff = policy_coeff
        DAgger.__init__(self, reference, policy, p_rollin_ref)

    def __call__(self, state):
        costs = np.zeros(1 + max(state.actions))
        self.reference.set_min_costs_to_go(state, costs)
        costs += self.policy_coeff * self.policy.predict_costs(state)
        ref = None
        # TODO vectorize then when |actions|=n_actions
        for a in state.actions:
            if ref is None or costs[a] < costs[ref]:
                ref = a
        pol = self.policy(state)
        self.objective += self.policy.forward(state, ref)
        if self.p_rollin_ref():
            return ref
        else:
            return pol


class TwistedDAgger(macarico.Learner):
    def __init__(self, reference, policy, p_rollin_ref):
        self.p_rollin_ref = p_rollin_ref
        self.policy = policy
        self.reference = reference
        self.objective = 0.0

#    @profile
    def __call__(self, state):
        ref = break_ties_by_policy(self.reference, self.policy, state, False)
        use_ref = self.p_rollin_ref()
        deviation = ref if use_ref else None
        pol = self.policy(state, deviate_to=deviation)
        self.objective += self.policy.forward(state, ref)
        return ref if use_ref else pol
        
#    @profile
    def update(self, _):
        self.objective.backward()
    
