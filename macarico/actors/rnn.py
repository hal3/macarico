from __future__ import division, generators, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
from torch.nn.parameter import Parameter

import macarico

def onehot(i):
    return Var(torch.LongTensor([int(i)]), requires_grad=False)

class RNNActor(macarico.Actor):
    def __init__(self,
                 attention,
                 n_actions,
                 d_actemb = 5, # TODO None = just BOW on actions
                 d_hid = 50,
                 cell_type = 'RNN', # RNN or LSTM or GRU
                ):
        macarico.Actor.__init__(self, d_hid, attention)

        assert cell_type in ['RNN', 'GRU', 'LSTM']
        
        self.d_actemb = d_actemb
        self.d_hid = d_hid
        self.cell_type = cell_type
        
        input_dim = d_actemb + sum((att.dim for att in self.attention))
        
        # TODO just make this an RNNBuilder
        self.embed_a = nn.Embedding(n_actions, self.d_actemb)
        self.rnn = getattr(nn, cell_type + 'Cell')(input_dim, self.d_hid)
        self.h = None

    def reset(self, env):
        self.h = Var(torch.zeros(1, self.d_hid), requires_grad=False)
        if self.cell_type == 'LSTM':
            self.h = self.h, Var(torch.zeros(1, self.d_hid), requires_grad=False)
        macarico.Actor.reset(self, env)

    def hidden(self):
        return self.h[0] if self.cell_type == 'LSTM' else self.h
        
    def _forward(self, state, x):
        # embed the previous action (if it exists)
        ae = Var(torch.zeros(1, self.d_actemb), requires_grad=False) \
             if len(state._trajectory) == 0 else \
             self.embed_a(onehot(state._trajectory[-1]))

        # combine prev hidden state, prev action embedding, and input x
        inputs = torch.cat([ae] + x, 1)
        self.h = self.rnn(inputs, self.h)
        return self.hidden()
    
