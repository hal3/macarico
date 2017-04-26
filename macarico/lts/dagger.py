from __future__ import division

import macarico

class DAgger(macarico.LearningAlg):

    def __init__(self, reference, policy, p_rollin_ref):
        self.p_rollin_ref = p_rollin_ref
        self.policy = policy
        self.reference = reference
        self.objective = 0.0

    def __call__(self, state, limit_actions=None):
        ref = self.reference(state, limit_actions=limit_actions)
        self.objective += self.policy.forward(state, ref, limit_actions=limit_actions)
        if self.p_rollin_ref():
            return ref
        else:
            return self.policy(state, limit_actions=limit_actions)

    def update(self, _):
        self.objective.backward()
