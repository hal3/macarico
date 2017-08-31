from __future__ import division

import random
import sys
#import torch
import numpy as np
import macarico

class AggreVaTe(macarico.Learner):

    def __init__(self, reference, policy, p_rollin_ref):        
        self.p_rollin_ref = p_rollin_ref
        self.policy = policy
        self.reference = reference
        self.objective = 0.0

    def __call__(self, state):
        pred_costs = self.policy.predict_costs(state)
        #from arsenal import ip; ip()
        costs = np.zeros(pred_costs.dim()[0][0]) # max(state.actions)+1)
        try:
            self.reference.set_min_costs_to_go(state, costs)
        except NotImplementedError:
            raise ValueError('can only run aggrevate on reference losses that define min_cost_to_go; try lols with rollout=ref instead')

        costs = costs - costs.min()
        
        ref = None
        for a in state.actions:
            if ref is None or costs[a] < costs[ref] or \
               (costs[a] == costs[ref] and pred_costs[a] < pred_costs[ref]):
                ref = a
        print costs
        self.objective += self.policy.forward_partial_complete(pred_costs, costs, state.actions)
        return ref if self.p_rollin_ref() else \
               self.policy.greedy(state, pred_costs)

    def update(self, _):
        self.objective.backward()
