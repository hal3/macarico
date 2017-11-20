from __future__ import division, generators, print_function

import numpy as np
import torch
import macarico
from macarico.annealing import stochastic, NoAnnealing
from macarico.util import break_ties_by_policy, argmin

class DAgger(macarico.Learner):
    def __init__(self, policy, reference, p_rollin_ref=NoAnnealing(0)):
        macarico.Learner.__init__(self)
        self.rollin_ref = stochastic(p_rollin_ref)
        self.policy = policy
        self.reference = reference
        self.objective = 0.0

    def forward(self, state):
        ref = break_ties_by_policy(self.reference, self.policy, state, False)
        pol = self.policy(state)
        self.objective += self.policy.update(state, ref)
        return ref if self.rollin_ref() else pol

    def update(self, _):
        obj = 0.0
        if not isinstance(self.objective, float):
            obj = self.objective.data[0]
            self.objective.backward()
        self.objective = 0.0
        self.rollin_ref.step()
        return obj


class Coaching(DAgger):
    def __init__(self, policy, reference, policy_coeff=0., p_rollin_ref=NoAnnealing(0)):
        DAgger.__init__(self, policy, reference, p_rollin_ref)
        self.policy_coeff = policy_coeff

    def forward(self, state):
        costs = torch.zeros(self.policy.n_actions)
        self.reference.set_min_costs_to_go(state, costs)
        pred_costs = self.policy.predict_costs(state)
        costs += self.policy_coeff * pred_costs.data
        ref = argmin(costs, state.actions)
        pol = self.policy(state)
        self.objective += self.policy.update(pred_costs, ref, state.actions)
        return ref if self.rollin_ref() else pol


