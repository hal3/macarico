from __future__ import division, generators, print_function

import torch
import torch.nn as nn

import macarico
import macarico.util as util
from macarico.util import Var, Varng

class BOWActor(macarico.Actor):
    def __init__(self, attention, n_actions, history_length=1):
        macarico.Actor.__init__(self,
                                history_length * n_actions + \
                                sum((att.dim for att in attention)),
                                attention)
        self.n_actions = n_actions
        self.history_length = history_length
        self._t = nn.Linear(1,1,bias=False) # need this so that if we get moved to GPU, we know
        
    def _forward(self, state, x):
        history = util.zeros(self._t.weight, 1, self.history_length * self.n_actions)
        for i in range(min(self.history_length, len(state._trajectory))):
            a = state._trajectory[-i]
            history[0, i * self.n_actions + a] = 1
        return torch.cat(x + [Varng(history)], 1)
