from __future__ import division, generators, print_function

import torch
import torch.nn as nn

import macarico
import macarico.util as util
from macarico.util import Var, Varng

class BOWActor(macarico.Actor):
    def __init__(self, attention, n_actions, act_history_length=1, obs_history_length=0):
        self.att_dim = sum((att.dim for att in attention))
        super().__init__(n_actions,
                         self.att_dim + 
                         act_history_length * n_actions + \
                         obs_history_length * self.att_dim,
                         attention)
        self.act_history_length = act_history_length
        self.obs_history_length = obs_history_length
        self._reset()

    def _forward(self, state, x):
        feats = x[:]
        if self.act_history_length > 0:
            f = util.zeros(self, 1, self.act_history_length * self.n_actions)
            for i in range(min(self.act_history_length, len(state._trajectory))):
                a = state._trajectory[-i]
                f[0, i * self.n_actions + a] = 1
            feats.append(Varng(f))
        if self.obs_history_length > 0:
            for i in range(self.obs_history_length):
                feats.append(Varng(self.obs_history[(self.obs_history_pos+i) % self.obs_history_length]))
            # update history
            self.obs_history[self.obs_history_pos] = torch.cat(x, dim=1).data
            self.obs_history_pos = (self.obs_history_pos + 1) % self.obs_history_length
        return torch.cat(feats, dim=1)

    def _reset(self):
        self.obs_history = []
        for _ in range(self.obs_history_length):
            self.obs_history.append(util.zeros(self, 1, self.att_dim))
        self.obs_history_pos = 0
        
