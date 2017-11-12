from __future__ import division
import torch
import random
import sys
import numpy as np
import macarico

class AggreVaTe(macarico.Learner):

    def __init__(self, reference, policy, p_rollin_ref, only_one_deviation=False):
        self.p_rollin_ref = p_rollin_ref
        self.policy = policy
        self.reference = reference
        self.objective = 0.0
        self.only_one_deviation = only_one_deviation
        self.t = None

    def __call__(self, state):
        if self.t is None:
            self.t = 0
            self.dev_t = np.random.choice(range(state.T)) + 1
        self.t += 1
        
        pred_costs = self.policy.predict_costs(state)
        costs = torch.zeros(max(state.actions)+1)
        try:
            self.reference.set_min_costs_to_go(state, costs)
        except NotImplementedError:
            raise ValueError('can only run aggrevate on reference losses that define min_cost_to_go; try lols with rollout=ref instead')

        costs = costs - costs.min()
        
        ref = None
        for a in state.actions:
            if ref is None or costs[a] < costs[ref] or \
               (costs[a] == costs[ref] and pred_costs.data[a] < pred_costs.data[ref]):
                ref = a
        if (not self.only_one_deviation) or (self.t == self.dev_t):
            self.objective += self.policy.forward_partial_complete(pred_costs, costs, state.actions)
        return ref if self.p_rollin_ref() else \
               self.policy.greedy(state, pred_costs)

    def update(self, _):
        if not isinstance(self.objective, float):
            self.objective.backward()
