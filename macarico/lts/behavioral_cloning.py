from __future__ import division, generators, print_function

import macarico

class BehavioralCloning(macarico.Learner):
    def __init__(self, policy, reference):
        macarico.Learner.__init__(self)
        #assert isinstance(policy, macarico.CostSensitivePolicy) # TODO HUH WHY DOESN"T THIS WORK?
        self.policy = policy
        self.reference = reference
        self.objective = 0.0

    def forward(self, state):
        ref = self.reference(state)
        self.objective += self.policy.update(state, ref)
        return ref

    def update(self, _):
        obj = 0.
        if not isinstance(self.objective, float):
            obj = self.objective.data[0]
            self.objective.backward()
        self.objective = 0.0
        return obj
