from __future__ import division, generators, print_function
import numpy as np
import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var

from macarico.annealing import EWMA, stochastic

import macarico


class Reinforce(macarico.Learner):
    "REINFORCE with a scalar baseline function."

    def __init__(self, policy, baseline=EWMA(0.8)):
        macarico.Learner.__init__(self)
        self.policy = policy
        self.baseline = baseline
        self.trajectory = []

    def update(self, loss):
        if len(self.trajectory) == 0: return

        b = 0 if self.baseline is None else self.baseline()
        total_loss = sum((torch.log(p_a) for p_a in self.trajectory)) * (loss - b)

        obj = total_loss.data[0]
        total_loss.backward()
        
        if self.baseline is not None:
            self.baseline.update(loss)

        self.trajectory = []
        return obj

    def forward(self, state):
        action, p_action = self.policy.stochastic_with_probability(state)
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
            x = Var(x.data)
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

    def update(self, loss):
        if len(self.trajectory) == 0: return
        loss = float(loss)
        loss_var = Var(torch.zeros(1) + loss, requires_grad=False)
        
        total_loss = 0.0
        for p_a, value in self.trajectory:
            v = value.data[0,0]

            # reinforcement loss
            total_loss += (loss - v) * p_a.log()

            # value fn approximator loss
            # TODO: loss should live in the VALUE, similar to policy
            total_loss += self.value_multiplier * self.loss_fn(value, loss_var)

        obj = total_loss.data[0]
        total_loss.backward()
        self.trajectory = []
        return obj

    def forward(self, state):
        action, p_action = self.policy.stochastic_with_probability(state)
        value = self.state_value_fn(state)
        # log action probabilities and values taken along current trajectory
        self.trajectory.append((p_action, value))
        return action
