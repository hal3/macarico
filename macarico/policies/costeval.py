from __future__ import division
import random
import dynet as dy
import numpy as np

from macarico import Policy
from macarico import util

class CostEvalPolicy(Policy):
    def __init__(self, reference, policy):
        self.policy = policy
        self.reference = reference
        self.costs = None
        self.n_actions = policy.n_actions
        self.record = [None] * 1000
        self.record_i = 0

    def __call__(self, state):
        if self.costs is None:
            self.costs = np.zeros(state.n_actions)

        self.costs *= 0
        self.reference.set_min_costs_to_go(state, self.costs)
        p_c = self.policy.predict_costs(state)
        self.record[self.record_i] = sum(abs(self.costs - p_c.npvalue()))
        self.record_i = (self.record_i + 1) % len(self.record)
        if np.random.random() < 1e-4 and self.record[-1] is not None:
            print sum(self.record) / len(self.record)
            #print sum(abs(self.costs - p_c.npvalue()))
            #print self.costs, p_c.npvalue()
        return self.policy.greedy(state, pred_costs=p_c)

    def predict_costs(self, state):
        return self.policy.predict_costs(state)

    def forward_partial_complete(self, pred_costs, truth, actions):
        return self.policy.forward_partial_complete(pred_costs, truth, actions)
    
    def update(self, _):
        pass

