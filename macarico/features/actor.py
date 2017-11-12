from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
from torch.nn.parameter import Parameter

import macarico

onehot = lambda i: Var(torch.LongTensor([i]), requires_grad=False)

"""
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
"""            

class RNNActor(macarico.Actor):
    def __init__(self,
                 attention,
                 n_actions,
                 d_actemb = 5, # TODO None = just BOW on actions
                 d_hid = 50,
                ):
        macarico.Actor.__init__(self, d_hid, attention)

        self.d_actemb = d_actemb
        self.d_hid = d_hid

        self.attention_oob = []
        attention_dim = 0
        for att_id, att in enumerate(self.attention):
            dim = att.features.dim
            oob = Parameter(torch.Tensor(att.arity or 1, dim))
            oob.data.zero_()
            self.register_parameter('attention_oob_%d' % att_id, oob)
            self.attention_oob.append(oob)
            attention_dim += dim * (att.arity or 1)

        input_dim = self.d_actemb + self.d_hid + attention_dim
        
        # TODO just make this an RNNBuilder
        self.embed_a = nn.Embedding(n_actions, self.d_actemb)
        self.combine = nn.Linear(input_dim, self.d_hid)
        self.h = None

    def reset(self, env):
        self.h = Var(torch.zeros(1, self.d_hid), requires_grad=False)
        macarico.Actor.reset(self, env)
        
    def compute(self, state, x):
        # embed the previous action (if it exists)
        ae = Var(torch.zeros(1, self.d_actemb), requires_grad=False) \
             if len(state._trajectory) == 0 else \
             self.embed_a(onehot(state._trajectory[-1]))

        # combine prev hidden state, prev action embedding, and input x
        inputs = torch.cat([self.h, ae] + x, 1)
        self.h = F.relu(self.combine(inputs))

        return self.h

"""
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
"""
