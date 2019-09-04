from __future__ import division, generators, print_function
import torch
import random
import sys
import numpy as np
import macarico
from macarico import CostSensitivePolicy
from macarico.annealing import NoAnnealing, stochastic
from macarico.util import break_ties_by_policy

class AggreVaTe(macarico.Learner):

    def __init__(self, policy, reference, p_rollin_ref=NoAnnealing(0)):
        macarico.Learner.__init__(self)
        assert isinstance(policy, CostSensitivePolicy)
        self.rollin_ref = stochastic(p_rollin_ref)
        self.policy = policy
        self.reference = reference
        self.objective = 0.0

    def forward(self, state):
        pred_costs = self.policy.predict_costs(state)
        costs = torch.zeros(self.policy.n_actions)
        try:
            costs.zero_()
            self.reference.set_min_costs_to_go(state, costs)
        except NotImplementedError:
            raise ValueError('can only run aggrevate on reference losses that define min_cost_to_go; try lols with rollout=ref instead')

        costs -= costs.min()
        self.objective += self.policy.update(pred_costs, costs, state.actions)
        
        return break_ties_by_policy(self.reference, self.policy, state, False) \
               if self.rollin_ref() else \
               self.policy.costs_to_action(state, pred_costs)

    def get_objective(self, _, final_state=None):
        ret = self.objective
        self.objective = 0.0
        self.rollin_ref.step()
        return ret
        
