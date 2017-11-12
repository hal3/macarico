from __future__ import division

import numpy as np
import macarico
from macarico.annealing import stochastic
from macarico.util import break_ties_by_policy

class DAgger(macarico.Learner):
    def __init__(self, reference, policy, p_rollin_ref):
        macarico.Learner.__init__(self)
        self.rollin_ref = stochastic(p_rollin_ref)
        self.policy = policy
        self.reference = reference
        self.objective = 0.0

    def forward(self, state):
        ref = break_ties_by_policy(self.reference, self.policy, state, False)
        pol = self.policy(state)
        self.objective += self.policy.forward(state, ref)
        return ref if self.rollin_ref() else pol

    def update(self, _):
        if not isinstance(self.objective, float):
            self.objective.backward()
        self.objective = 0.0
        self.rollin_ref.step()


class Coaching(DAgger):
    def __init__(self, reference, policy, p_rollin_ref, policy_coeff=0.):
        self.policy_coeff = policy_coeff
        DAgger.__init__(self, reference, policy, p_rollin_ref)

    def __call__(self, state):
        costs = torch.zeros(1 + max(state.actions))
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
    
