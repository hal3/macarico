from __future__ import division, generators, print_function

import torch
from torch.autograd import Variable as Var

import macarico

class BOWActor(macarico.Actor):
    def __init__(self, attention, n_actions, history_length=1):
        macarico.Actor.__init__(self,
                                history_length * n_actions + \
                                sum((att.dim for att in attention)),
                                attention)
        self.n_actions = n_actions
        self.history_length = history_length
        
    def _forward(self, state, x):
        history = torch.zeros(1, self.history_length * self.n_actions)
        for i in range(min(self.history_length, len(state._trajectory))):
            a = state._trajectory[-i]
            history[0, i * self.n_actions + a] = 1
        return torch.cat(x + [Var(history, requires_grad=False)], 1)
