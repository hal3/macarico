from __future__ import division

import macarico

class MaximumLikelihood(macarico.LearningAlg):

    def __init__(self, reference, policy):
        self.policy = policy
        self.reference = reference
        self.objective = 0.0

    def __call__(self, state, limit_actions=None):
        a = self.reference(state, limit_actions=limit_actions)
        self.objective += self.policy.forward(state, a, limit_actions=limit_actions)
        return a

    def update(self, _):
        self.objective.backward()
