from __future__ import division, generators, print_function

import macarico

class MaximumLikelihood(macarico.Learner):

    def __init__(self, reference, policy):
        self.policy = policy
        self.reference = reference
        self.objective = 0.0
        self.squared_loss = 0.

    def __call__(self, state):
        a = self.reference(state)
        self.objective += self.policy.forward(state, a)
        self.squared_loss = self.objective.data[0]
        return a

    def update(self, _):
        self.objective.backward()
