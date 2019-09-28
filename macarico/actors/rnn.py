import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import macarico
import macarico.util as util
from macarico.util import Var, Varng

class RNNActor(macarico.Actor):
    def __init__(self,
                 attention,
                 n_actions,
                 d_hid = 50,
                 d_actemb = None, # None = just BOW on actions
                 cell_type = 'LSTM', # RNN or LSTM or GRU
                ):
        super().__init__(n_actions, d_hid, attention)

        assert cell_type in ['RNN', 'GRU', 'LSTM']

        self.d_actemb = d_actemb
        self.d_hid = d_hid
        self.cell_type = cell_type
        
        input_dim = (d_actemb or (1+n_actions)) + sum((att.dim for att in self.attention))
        
        if self.d_actemb is not None:
            self.embed_a = nn.Embedding(1+n_actions, self.d_actemb)
        self.rnn = getattr(nn, cell_type + 'Cell')(input_dim, self.d_hid)
        self.h = None

    def _reset(self):
        self.h = Varng(util.zeros(self.rnn.weight_ih, 1, self.d_hid))
        if self.cell_type == 'LSTM':
            self.h = self.h, Varng(util.zeros(self.rnn.weight_ih, 1, self.d_hid))

    def hidden(self):
        return self.h[0] if self.cell_type == 'LSTM' else self.h
        
    def _forward(self, state, x):
        w = self.rnn.weight_ih
        # embed the previous action (if it exists)
        last_a = self.n_actions if len(state._trajectory) == 0 else state._trajectory[-1]
        if self.d_actemb is None:
            prev_a = util.zeros(w, 1, 1+self.n_actions)
            prev_a[0,last_a] = 1
            prev_a = Varng(prev_a)
        else:
            prev_a = self.embed_a(util.onehot(w, last_a))

        # combine prev hidden state, prev action embedding, and input x
        inputs = torch.cat([prev_a] + x, 1)
        self.h = self.rnn(inputs, self.h)
        return self.hidden()
    
