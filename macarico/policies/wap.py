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
from torch.autograd import Variable as Var


from macarico import Policy
from macarico import util

class WAPPolicy(Policy):
    """Linear policy, with weighted all pairs

    Notes:

    This policy can be trained with
    - policy gradient via `policy.stochastic().reinforce(reward)`

    - Cost-sensitive one-against-all linear regression (CSOAA) via
      `policy.forward(state, truth)`

    """

    def __init__(self,features, n_actions):
        
        self.n_actions = n_actions
        dim = 1 if features is None else features.dim
        n_actions_choose_2 = (n_actions * (n_actions-1)) // 2
        self._wap_w = dy_model.add_parameters((n_actions_choose_2, dim))
        self._wap_b = dy_model.add_parameters(n_actions_choose_2)
        self.features = features

    def __call__(self, state):
        return self.greedy(state)

    def sample(self, state):
        return self.stochastic(state)

    def stochastic(self, state, temperature=1):
        return self.stochastic_with_probability(state, temperature)[0]

    def stochastic_with_probability(self, state, temperature=1):
        assert False
#        p = self.predict_costs(state)
#        if len(state.actions) != self.n_actions:
#            disallow = torch.zeros(self.n_actions)
#            for i in range(self.n_actions):
#                if i not in state.actions:
#                    disallow[i] = 1e10
#            p += dy.inputTensor(disallow)
#        probs = dy.softmax(- p / temperature)
#        return util.sample_from_probs(probs)
    
#    @profile
    def predict_costs(self, state):
        "Predict costs using the csoaa model accounting for `state.actions`"
        if self.features is None:
            feats = dy.parameter(self.dy_model.add_parameters(1))
            self.features = lambda _: feats
        else:
            feats = self.features(state)

        wap_w = dy.parameter(self._wap_w)
        wap_b = dy.parameter(self._wap_b)
        return dy.affine_transform([wap_b, wap_w, feats])

#    @profile
    def greedy(self, state, pred_costs=None):
        if pred_costs is None:
            pred_costs = self.predict_costs(state)
        if isinstance(pred_costs, dy.Expression):
            pred_costs = pred_costs.data
        costs = torch.zeros(self.n_actions)
        k = 0
        for i in xrange(self.n_actions):
            for j in xrange(i):
                costs[i] -= pred_costs[k]
                costs[j] += pred_costs[k]
                k += 1
        if len(state.actions) == self.n_actions:
            return costs.argmin()
        best = None
        for a in state.actions:
            if best is None or costs[a] < costs[best]:
                best = a
        return best

#    @profile
    def forward_partial_complete(self, pred_costs, truth, actions):
        if isinstance(truth, int):
            truth = [truth]
        if isinstance(truth, list):
            truth0 = truth
            truth = torch.ones(self.n_actions)
            for k in truth0:
                truth[k] = 0.
        obj = 0.
        k = 0
        if not isinstance(actions, set):
            actions = set(actions)
        for i in xrange(self.n_actions):
            for j in xrange(i):
                weight = abs(truth[i] - truth[j])
                label = -1 if truth[i] > truth[j] else +1
                if weight > 1e-6:
                    #l = max(0, 1 -  label * pred_costs[k])
                    l = 1 - label * pred_costs[k]
                    l = 0.5 * (l + dy.abs(l)) # hinge
                    obj += weight * l
                k += 1
#        for a in actions:
#            v = (pred_costs[a] - truth[a])
#            obj += 0.5 * v * v
        return obj

#    @profile
    def forward(self, state, truth):
        costs = self.predict_costs(state)
        return self.forward_partial_complete(costs, truth, state.actions)
