from __future__ import division, generators, print_function

import macarico

class BehavioralCloning(macarico.Learner):
    def __init__(self, policy, reference):
        macarico.Learner.__init__(self)
        self.policy = policy
        self.reference = reference
        self.objective = 0.0

    def forward(self, state):
        ref = self.reference(state)
        self.objective += self.policy.forward(state, ref)
        return ref

    def update(self, _):
        if not isinstance(self.objective, float):
            self.objective.backward()
        self.objective = 0.0
