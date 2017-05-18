from __future__ import division

import macarico
from macarico.util import break_ties_by_policy

class DAgger(macarico.LearningAlg):

    def __init__(self, reference, policy, p_rollin_ref):
        self.p_rollin_ref = p_rollin_ref
        self.policy = policy
        self.reference = reference
        self.objective = 0.0

    def __call__(self, state):
        ref = self.reference(state) # break_ties_by_policy(self.reference, self.policy, state, False)
        pol = self.policy(state)
        self.objective += self.policy.forward(state, ref)
        if self.p_rollin_ref():
            return ref
        else:
            return pol

    def update(self, _):
        self.objective.backward()
