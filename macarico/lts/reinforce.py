import numpy as np
import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from macarico.annealing import EWMA, stochastic
from macarico.util import Var, Varng

import macarico
from macarico import StochasticPolicy

class Reinforce(macarico.Learner):
    "REINFORCE with a scalar baseline function."

    def __init__(self, policy, baseline=EWMA(0.8)):
        macarico.Learner.__init__(self)
        assert isinstance(policy, StochasticPolicy)
        self.policy = policy
        self.baseline = baseline
        self.trajectory = []

    def get_objective(self, loss, final_state=None):
        if len(self.trajectory) == 0: return 0.

        b = 0 if self.baseline is None else self.baseline()
        total_loss = sum((torch.log(p_a) for p_a in self.trajectory)) * (loss - b)

        if self.baseline is not None:
            self.baseline.update(loss)

        self.trajectory = []
        return total_loss

    def forward(self, state):
        action, p_action = self.policy.stochastic(state)
        self.trajectory.append(p_action)
        return action

class LinearValueFn(nn.Module):
    def __init__(self, features, disconnect_values=True):
        nn.Module.__init__(self)
        self.features = features
        self.dim = features.dim
        self.disconnect_values = disconnect_values
        self.value_fn = nn.Linear(self.dim, 1)

    def forward(self, state):
        x = self.features(state)
        if self.disconnect_values:
            x = Varng(x.data)
        #x *= 0
        #x[0,0] = 1
        return self.value_fn(x)


class A2C(macarico.Learner):
    def __init__(self, policy, state_value_fn, value_multiplier=1.0):
        macarico.Learner.__init__(self)
        self.policy = policy
        self.state_value_fn = state_value_fn
        self.trajectory = []
        self.value_multiplier = value_multiplier
        self.loss_fn = nn.SmoothL1Loss()
        self.loss_var = torch.zeros(1)

    def get_objective(self, loss, final_state=None):
        if len(self.trajectory) == 0: return
        loss = float(loss)
        loss_var = Varng(self.loss_var + loss)
        
        total_loss = 0.0
        for p_a, value in self.trajectory:
            v = value.data[0,0]

            # reinforcement loss
            total_loss += (loss - v) * p_a.log()

            # value fn approximator loss
            total_loss += self.value_multiplier * self.loss_fn(value, loss_var)

        self.trajectory = []
        return total_loss

    def forward(self, state):
        action, p_action = self.policy.stochastic(state)
        value = self.state_value_fn(state)
        # log action probabilities and values taken along current trajectory
        self.trajectory.append((p_action, value))
        return action
