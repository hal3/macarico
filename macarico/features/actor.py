from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
from torch.nn.parameter import Parameter

import macarico

onehot = lambda i: Var(torch.LongTensor([i]), requires_grad=False)

def initialize_subfeatures(model, sub_features, foci):
    model.sub_features = {}
    for sub_num, f in enumerate(sub_features):
        if f.field in model.sub_features:
            raise ValueError('multiple feature functions using same output field "%s"' % f.field)
        model.sub_features[f.field] = f
        #if isinstance(f, nn.Module):
        #    model.add_module('subfeatures_%d' % sub_num, f)

    model.foci = foci
    model.foci_dim = 0
    for foci_num, focus in enumerate(model.foci):
        if focus.field not in model.sub_features:
            raise ValueError('focus asking for field "%s" but this does not exist in the constructed sub-features' % focus.field)
        model.foci_dim += model.sub_features[focus.field].dim * (focus.arity or 1)
        #if isinstance(focus, nn.Module):
        #    model.add_module('focus_%d' % foci_num, focus)
            

class TransitionRNN(macarico.Features):

    def __init__(self,
                 sub_features,
                 foci,
                 n_actions,
                 d_actemb = 5,
                 d_hid = 50,
                 h_name = 'h',
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
        self.h_name = h_name

        # TODO just make this an RNNBuilder
        
        initialize_subfeatures(self, sub_features, foci)

        # focus model; compute dimensionality
        self.foci_oob = []
        for foci_num, focus in enumerate(self.foci):
            dim = self.sub_features[focus.field].dim
            oob_param = Parameter(torch.Tensor(focus.arity or 1, dim))
            self.register_parameter('foci_oob_%d' % foci_num, oob_param)
            oob_param.data.zero_()
            self.foci_oob.append(oob_param)

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

        macarico.Features.__init__(self, None, self.d_hid)

#    @profile
    def forward(self, state):
        t = state.t

        if not hasattr(state, self.h_name) or getattr(state, self.h_name) is None or t == 0:
            setattr(state, self.h_name, [None]*state.T)
            setattr(state, self.h_name + '0', self.initial_h)
            
        h = getattr(state, self.h_name)
        if h[t] is not None:
            return h[t]
        
        if t == 0:
            prev_h = self.initial_h
            ae = self.initial_ae
        else:
            prev_h = h[t-1].view(1, self.d_hid)
            # embed the previous action (if it exists)
            if len(state.output) >= t:
                ae = self.embed_a(onehot(state.output[t-1]))
            else:
                ae = self.initial_ae

        # Combine input embedding, prev hidden state, and prev action embedding
        inputs = [ae, prev_h]
        cached_sub_features = {}
        for foci_num, focus in enumerate(self.foci):
            idx = focus(state)
            # TODO: try to generaize the two branches below
            if focus.arity is not None:
                assert len(idx) == focus.arity, \
                    'focus %s is lying about its arity (claims %d, got %s)' % \
                    (focus, focus.arity, idx)
                if focus.field not in cached_sub_features:
                    cached_sub_features[focus.field] = self.sub_features[focus.field](state)  # 40% of time (predict/train)
                feats = cached_sub_features[focus.field]
#                feats = self.sub_features[focus.field](state)
                for idx_num, i in enumerate(idx):
                    if i is None:
                        #inputs.append(zeros(self.sub_features[focus.field].dim))
                        oob = self.foci_oob[foci_num][idx_num]
                        inputs.append(oob.view(1, self.sub_features[focus.field].dim))  # TODO move the .view to the construction of oob
                    else:
                        inputs.append(feats[i])
            else:  # focus.arity is None
                feats = self.sub_features[focus.field](state)
                assert idx.data.shape[1] == feats.data.shape[0], \
                    'focus %s returned something of the wrong size (returned %s, needed %s)' % \
                    (focus, idx.dim(), feats)
                #print 'idx.size =', idx.size()
                #print 'feats.size =', feats.squeeze(1).size()
                inputs.append(torch.mm(idx, feats.squeeze(1)))

        h[t] = F.relu(self.combine(torch.cat(inputs, 1)))

        return h[t]

    def deviate_by(self, state, dev):
        t = state.t
        h = getattr(state, self.h_name)
        h[t] += Var(dev, requires_grad=False)

class TransitionBOW(macarico.Features):
    def __init__(self,sub_features, foci, n_actions, max_length=255):
        nn.Module.__init__(self)
        
        self.sub_features = sub_features
        self.foci = foci
        self.n_actions = n_actions

        initialize_subfeatures(self, sub_features, foci)
        self.dim = self.foci_dim + self.n_actions
        self.zeros = {}
        for focus in self.sub_features.itervalues():
            if focus.dim not in self.zeros:
                self.zeros[focus.dim] = Var(torch.zeros((1, focus.dim)), requires_grad=False)
        
        macarico.Features.__init__(self, None, self.dim)
                 
#    @profile
    def forward(self, state):
        t = state.t

        inputs = []
        cached_sub_features = {}
        for focus in self.foci:
            idx = focus(state)
            if focus.field not in cached_sub_features:
                cached_sub_features[focus.field] = self.sub_features[focus.field](state)  # 40% of time (predict/train)
            feats = cached_sub_features[focus.field]
            for i in idx:
                if i is None:
                    inputs.append(self.zeros[self.sub_features[focus.field].dim])
                else:
                    inputs.append(feats[i])

        action = torch.zeros(1, self.n_actions)
        if len(state.output) > 0:
            a = state.output[-1]
            if a >= 0 and a < self.n_actions:
                action[0, a] = 1.
        inputs.append(Var(action, requires_grad=False))
        return torch.cat(inputs, 1)
