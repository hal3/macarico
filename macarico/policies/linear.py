from __future__ import division, generators, print_function
import random
#import torch
#from torch import nn
#from torch.autograd import Variable
#from torch.nn import functional as F
#from torch.nn.parameter import Parameter
#import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from macarico.util import Var, Varng
from torch.nn.parameter import Parameter
import numpy as np

import macarico
from macarico import util, CostSensitivePolicy

class SoftmaxPolicy(macarico.StochasticPolicy):
    def __init__(self, features, n_actions, temperature=1.0):
        macarico.StochasticPolicy.__init__(self)
        self.n_actions = n_actions
        self.features = features
        self.mapping = nn.Linear(features.dim, n_actions)
        self.disallow = torch.zeros(n_actions)
        self.temperature = temperature

    def forward(self, state):
        z = self.mapping(self.features(state)).squeeze()
        if len(state.actions) == self.n_actions:
            _, i = z.min(0)
            return i.data[0]
        i = None
        for a in state.actions:
            if i is None or z[a] < z[i]:
                i = a
        return i

    def stochastic(self, state):
        z = self.mapping(self.features(state)).squeeze()
        if len(state.actions) != self.n_actions:
            self.disallow.zero_()
            self.disallow += 1e10
            for a in state.actions:
                self.disallow[a] = 0.
            z += Varng(self.disallow)
        p = F.softmax(-z / self.temperature, dim=0)
        return util.sample_from_probs(p)

def truth_to_vec(truth, tmp_vec):
    if isinstance(truth, torch.FloatTensor):
        return truth
    if isinstance(truth, int):
        tmp_vec.zero_()
        tmp_vec += 1
        tmp_vec[truth] = 0
        return tmp_vec
    if isinstance(truth, list) or isinstance(truth, set):
        tmp_vec.zero_()
        tmp_vec += 1
        for t in truth:
            tmp_vec[t] = 0
        return tmp_vec
    raise ValueError('invalid argument type for "truth", must be in, list or set')
    
class CSOAAPolicy(SoftmaxPolicy, CostSensitivePolicy):
    def __init__(self, features, n_actions, loss_fn='huber', temperature=1.0, clamp_costs=True, min_cost=None, max_cost=None):
        SoftmaxPolicy.__init__(self, features, n_actions, temperature)
        self.clamp_costs = clamp_costs
        self.min_cost = min_cost
        self.max_cost = max_cost
        self.set_loss(loss_fn)

    def set_loss(self, loss_fn):
        assert loss_fn in ['squared', 'huber']
        self.loss_fn = nn.MSELoss(size_average=False)      if loss_fn == 'squared' else \
                       nn.SmoothL1Loss(size_average=False) if loss_fn == 'huber' else \
                       None
    
    def predict_costs(self, state):
        z = self.mapping(self.features(state)).squeeze()
        if self.clamp_costs and self.min_cost is not None:
            z = z.clamp(self.min_cost, self.max_cost)
        return z

    def _compute_loss(self, loss_fn, pred_costs, truth, state_actions):
        if len(state_actions) == self.n_actions:
            return loss_fn(pred_costs, Varng(truth))
        return sum((loss_fn(pred_costs[a], Varng(torch.zeros(1) + truth[a])) \
                    for a in state_actions))

    def _update_cost_range(self, truth, actions=None): # TODO use actions
        if self.clamp_costs:
            mn, mx = truth.min(), truth.max()
            if self.min_cost is None or mn < self.min_cost: self.min_cost = mn
            if self.max_cost is None or mx > self.max_cost: self.max_cost = mx
    
    def _update(self, pred_costs, truth, actions=None):
        truth = truth_to_vec(truth, torch.zeros(self.n_actions))
        self._update_cost_range(truth, actions)
        return self._compute_loss(self.loss_fn, pred_costs, truth, actions)

class WMCPolicy(CSOAAPolicy):
    def __init__(self, features, n_actions, loss_fn='hinge', temperature=1.0, clamp_costs=False, min_cost=None, max_cost=None):
        CSOAAPolicy.__init__(self, features, n_actions, loss_fn, temperature, clamp_costs, min_cost, max_cost)
        
    def set_loss(self, loss_fn):
        assert loss_fn in ['multinomial', 'hinge', 'squared', 'huber']
        if loss_fn == 'hinge':
            l = nn.MultiMarginLoss(size_average=False)
            self.loss_fn = lambda p, t, _: l(p, Varng(torch.LongTensor([t])))
        elif loss_fn == 'multinomial':
            l = nn.NLLLoss(size_average=False)
            self.loss_fn = lambda p, t, _: l(F.log_softmax(p.unsqueeze(0), dim=1), Varng(torch.LongTensor([t])))
        elif loss_fn in ['squared', 'huber']:
            l = (nn.MSELoss if loss_fn == 'squared' else nn.SmoothL1Loss)(size_average=False)
            self.loss_fn = lambda p, t, sa: self._compute_loss(l, p, 1 - truth_to_vec(t, torch.zeros(self.n_actions)), sa)
        
    def _update(self, pred_costs, truth, actions=None):
        truth = truth_to_vec(truth, torch.zeros(self.n_actions))
        self._update_cost_range(truth, actions)
        pred_costs = -pred_costs
        
        if len(actions) <= 1:
            return 0. # TODO maybe still make an example
        
        # TODO if truth is just an int, we can shortcut a lot of this
        w = truth.sum() / (len(actions)-1) - truth
        w -= w.min()
        obj = 0.
        for a in actions:
            # set up a classification example where a is the correct
            # class with importance weight w[a], where w[a] = (sum
            # costs)/(K-1) - c[a], centered so minimum is zero (and
            # therefore highest cost action doesn't get a positive
            # example)
            if w[a] <= 1e-6: continue
            obj += self.loss_fn(pred_costs, a, actions)
        return obj
            
