from __future__ import division

import macarico

class MaximumLikelihood(macarico.LearningAlg):

    def __init__(self, reference, policy, p_rollin_ref):
        self.p_rollin_ref = p_rollin_ref
        self.policy = policy
        self.reference = reference
        self.objective = 0.0

    def __call__(self, state):
        a = self.reference(state)
        self.objective += self.policy.forward(state, a)
        return a

    def update(self, _):
        self.objective.backward()
