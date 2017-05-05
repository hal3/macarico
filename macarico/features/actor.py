from __future__ import division

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable

import macarico

zeros = lambda d: Variable(torch.zeros(1,d))
onehot = lambda i: Variable(torch.LongTensor([i]))

class TransitionRNN(macarico.Features, nn.Module):

    def __init__(self,
                 sub_features,
                 foci,
                 n_actions,
                 d_actemb = 5,
                 d_hid = 50,
                ):
        nn.Module.__init__(self)

        # model is:
        #   h[-1] = zero
        #   for n in xrange(N):
        #     ae   = embed_action(y[n-1]) or zero if n=0
        #     h[n] = combine([f for f in foci], ae, h[n-1])
        #     y[n] = act(h[n])
        # we need to know:
        #   d_hid     - hidden state
        #   d_actemb  - action embeddings

        self.d_actemb = d_actemb
        self.d_hid = d_hid
        self.sub_features = {}
        for f in sub_features:
            field = f.output_field
            if field in self.sub_features:
                raise ValueError('multiple feature functions using same output field "%s"' % field)
            self.sub_features[field] = f

        # focus model; compute dimensionality
        self.foci = foci
        self.foci_oob = []
        input_dim = self.d_actemb + self.d_hid
        for foci_num, focus in enumerate(self.foci):
            if focus.field not in self.sub_features:
                raise ValueError('focus asking for field "%s" but this does not exist in the constructed sub-features' % focus.field)
            dim = self.sub_features[focus.field].dim
            input_dim += focus.arity * dim
            oob_param = Parameter(torch.Tensor(focus.arity, dim))
            self.foci_oob.append(oob_param)
            self.register_parameter('foci_oob_%d' % foci_num, oob_param)

        # nnet models
        self.embed_a = nn.Embedding(n_actions, self.d_actemb)
        self.combine = nn.Linear(input_dim, self.d_hid)
        self.initial_h = Parameter(torch.Tensor(1,self.d_hid))

        macarico.Features.__init__(self, self.d_hid)

    def forward(self, state):
        t = state.t

        if not hasattr(state, 'h') or state.h is None:
            state.h = [None]*state.T
            prev_h = self.initial_h # Variable(torch.zeros(1, self.d_hid))
            ae = zeros(self.d_actemb)
        else:
            if state.h[t] is not None:
                return state.h[t]

            prev_h = state.h[t-1].resize(1, self.d_hid)
            # embed the previous action (if it exists)
            ae = self.embed_a(onehot(state.output[t-1]))

        # Combine input embedding, prev hidden state, and prev action embedding
        #inputs = [state.r[i] if i is not None else zeros(self.d_rnn*2) for i in self.foci(state)] + [ae, prev_h]
        inputs = [ae, prev_h]
        for foci_num, focus in enumerate(self.foci):
            idx = focus(state)
            assert len(idx) == focus.arity, \
                'focus %s is lying about its arity (claims %d, got %s)' % \
                (focus, focus.arity, idx)
            feats = self.sub_features[focus.field](state)
            for idx_num,i in enumerate(idx):
                if i is None:
                    #inputs.append(zeros(self.sub_features[focus.field].dim))
                    oob = self.foci_oob[foci_num][idx_num,:]
                    inputs.append(oob.resize(1, self.sub_features[focus.field].dim))
                else:
                    inputs.append(feats[i])

        state.h[t] = F.tanh(self.combine(torch.cat(inputs, 1)))

        return state.h[t]


