from __future__ import division

import random
import sys
import torch
from torch.autograd import Variable
import macarico

class AggreVaTe(macarico.LearningAlg):

    def __init__(self, reference, policy, p_rollin_ref):
        if not hasattr(reference, 'min_cost_to_go'):
            raise ValueError('can only run aggrevate on reference losses that define min_cost_to_go; try lols with rollout=ref instead')
        
        self.p_rollin_ref = p_rollin_ref
        self.policy = policy
        self.reference = reference
        self.objective = 0.0

    def __call__(self, state):
        ref = self.reference(state)
        pol = self.policy(state)        
        costs = torch.zeros(1, max(state.actions)+1)
        for a in state.actions:
            costs[0,a] = self.reference.min_cost_to_go(state, a)
        self.objective += self.policy.forward(state, costs)
        return ref if self.p_rollin_ref() \
               else pol

    def update(self, _):
        self.objective.backward()
