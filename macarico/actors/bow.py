from __future__ import division, generators, print_function

import torch
from torch.autograd import Variable as Var

import macarico

class BOWActor(macarico.Actor):
    def __init__(self, attention, n_actions, max_length=255):
        macarico.Actor.__init__(self, n_actions + sum((att.dim for att in attention)), attention)
        self.n_actions = n_actions
        
    def _forward(self, state, x):
        action = torch.zeros(1, self.n_actions)
        if len(state._trajectory) > 0:
            action[0, state._trajectory[-1]] = 1
        return torch.cat(x + [Var(action, requires_grad=False)], 1)
