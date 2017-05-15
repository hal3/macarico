from __future__ import division

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable

import macarico

#zeros = lambda d: Variable(torch.zeros(1,d))
onehot = lambda i: Variable(torch.LongTensor([i]), requires_grad=False)

def initialize_subfeatures(model, sub_features, foci):
    model.sub_features = {}
    for f in sub_features:
        field = f.output_field
        if field in model.sub_features:
            raise ValueError('multiple feature functions using same output field "%s"' % field)
        model.sub_features[field] = f

    model.foci = foci
    model.foci_dim = 0
    for foci_num, focus in enumerate(model.foci):
        if focus.field not in model.sub_features:
            raise ValueError('focus asking for field "%s" but this does not exist in the constructed sub-features' % focus.field)
        model.foci_dim += model.sub_features[focus.field].dim * focus.arity

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

        initialize_subfeatures(self, sub_features, foci)

        # focus model; compute dimensionality
        self.foci_oob = []
        for foci_num, focus in enumerate(self.foci):
            dim = self.sub_features[focus.field].dim
            oob_param = Parameter(torch.Tensor(focus.arity, dim))
            self.foci_oob.append(oob_param)
            self.register_parameter('foci_oob_%d' % foci_num, oob_param)
            oob_param.data.zero_()

        # nnet models
        input_dim = self.d_actemb + self.d_hid + self.foci_dim
        
        self.embed_a = nn.Embedding(n_actions, self.d_actemb)
        self.combine = nn.Linear(input_dim, self.d_hid)
        initial_h_tensor = torch.Tensor(1,self.d_hid)
        initial_h_tensor.zero_()
        self.initial_h = Parameter(initial_h_tensor)
        initial_ae_tensor = torch.Tensor(1,self.d_actemb)
        initial_ae_tensor.zero_()
        self.initial_ae = Parameter(initial_ae_tensor)

        macarico.Features.__init__(self, self.d_hid)

    def forward(self, state):
        t = state.t

        if not hasattr(state, 'h') or state.h is None:
            state.h = [None]*state.T

        if state.h[t] is not None:
            return state.h[t]
        
        if t == 0:
            prev_h = self.initial_h
            #prev_h = Variable(torch.zeros(1, self.d_hid))
            ae = self.initial_ae
        else:
            prev_h = state.h[t-1].resize(1, self.d_hid)
            # embed the previous action (if it exists)
            ae = self.embed_a(onehot(state.output[t-1]))

        # Combine input embedding, prev hidden state, and prev action embedding
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


class TransitionBOW(macarico.Features, nn.Module):

    def __init__(self, sub_features, foci, n_actions, max_length=255):
        nn.Module.__init__(self)

        self.sub_features = sub_features
        self.foci = foci
        self.n_actions = n_actions

        initialize_subfeatures(self, sub_features, foci)
        self.dim = self.foci_dim + self.n_actions
        #self.features = Variable(torch.zeros(max_length, 1, self.dim),
        #                         requires_grad=False)
        self.zeros = {}
        for focus in self.sub_features.itervalues():
            if focus.dim not in self.zeros:
                self.zeros[focus.dim] = Variable(torch.zeros(1, focus.dim), requires_grad=False)
        
        macarico.Features.__init__(self, self.dim)
                 
    #@profile
    def forward(self, state):
        t = state.t

        inputs = []
        cached_sub_features = {}
        for focus in self.foci:
            idx = focus(state)
            if focus.field not in cached_sub_features:
                cached_sub_features[focus.field] = self.sub_features[focus.field](state)  # 40% of time (predict/train)
            feats = cached_sub_features[focus.field]
            for idx_num, i in enumerate(idx):
                if i is None:
                    #prev = torch.zeros(1, self.sub_features[focus.field].dim)
                    inputs.append(self.zeros[self.sub_features[focus.field].dim])
                else:
                    inputs.append(feats[i])

        action = torch.zeros(1, self.n_actions)
        if len(state.output) > 0:
            a = state.output[-1]
            if a >= 0 and a < self.n_actions:
                action[0,a] = 1.
        inputs.append(Variable(action, requires_grad=False))
                    
        return torch.cat(inputs, 1)   # 30% of time (predict/train)
    
    
