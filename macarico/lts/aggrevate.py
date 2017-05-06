from __future__ import division

import random
import sys
import torch
from torch.autograd import Variable
import macarico

class AggreVaTe(macarico.LearningAlg):

    def __init__(self, reference, policy, p_rollin_ref, break_ties_by_policy=True):
        if not hasattr(reference, 'min_cost_to_go'):
            raise ValueError('can only run aggrevate on reference losses that define min_cost_to_go; try lols with rollout=ref instead')
        
        self.p_rollin_ref = p_rollin_ref
        self.policy = policy
        self.reference = reference
        self.objective = 0.0
        self.break_ties_by_policy = break_ties_by_policy

    def __call__(self, state):
        pred_costs = self.policy.predict_costs(state)
        costs = torch.zeros(1, max(state.actions)+1) + 1e10
        best_cost = None
        best_pred = None
        for a in state.actions:
            cost = self.reference.min_cost_to_go(state, a)
            costs[0,a] = cost
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_pred = a
            elif cost == best_cost and pred_costs[0,a] < pred_costs[0,best_pred]:
                best_pred = a
        ref = best_pred if self.break_ties_by_policy else self.reference(state)
        self.objective += self.policy.forward_partial_complete(pred_costs, costs, state.actions)
        return ref if self.p_rollin_ref() else \
               self.policy.greedy(state, pred_costs)

    def update(self, _):
        self.objective.backward()
