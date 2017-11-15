from __future__ import division, generators, print_function
import torch
import random
import sys
import numpy as np
import macarico
from macarico.annealing import NoAnnealing, stochastic
from macarico.util import break_ties_by_policy

class AggreVaTe(macarico.Learner):

    def __init__(self, policy, reference, p_rollin_ref=NoAnnealing(0)):
        macarico.Learner.__init__(self)
        
        self.rollin_ref = stochastic(p_rollin_ref)
        self.policy = policy
        self.reference = reference
        self.objective = 0.0
        self.costs = torch.zeros(policy.n_actions)

    def forward(self, state):
        pred_costs = self.policy.predict_costs(state)
        try:
            self.costs.zero_()
            self.reference.set_min_costs_to_go(state, self.costs)
        except NotImplementedError:
            raise ValueError('can only run aggrevate on reference losses that define min_cost_to_go; try lols with rollout=ref instead')

        self.costs = self.costs - self.costs.min()
        self.objective += self.policy.forward_partial_complete(pred_costs, self.costs, state.actions)
        
        return break_ties_by_policy(self.reference, self.policy, state, False) \
               if self.rollin_ref() else \
               self.policy.greedy(state, pred_costs)

    def update(self, _):
        obj = 0
        if not isinstance(self.objective, float):
            obj = self.objective.data[0]
            self.objective.backward()
        self.objective = 0.0
        self.rollin_ref.step()
        return obj
        
