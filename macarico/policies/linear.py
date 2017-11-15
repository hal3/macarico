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
from macarico import util

class LinearPolicy(macarico.Policy):
    """Linear policy

    Notes:

    This policy can be trained with
    - policy gradient via `policy.stochastic().reinforce(reward)`

    - Cost-sensitive one-against-all linear regression (CSOAA) via
      `policy.forward(state, truth)`

    """

    def __init__(self, features, n_actions, loss_fn='huber'):
        macarico.Policy.__init__(self)

        self.n_actions = n_actions
        dim = 1 if features is None else features.dim

        self.mapping = nn.Linear(dim, n_actions)

        if   loss_fn == 'squared': self.distance = nn.MSELoss(size_average=False)
        elif loss_fn == 'huber':   self.distance = nn.SmoothL1Loss(size_average=False)
        else: assert False, ('unknown loss function %s' % loss_fn)
        
        self.features = features
        self.disallow = torch.zeros(n_actions)
        self.unit = torch.zeros(1)

    def sample(self, state):
        return self.stochastic(state)
        #return self.stochastic(state).view(1)[0]   # get an integer instead of pytorch.variable

#    @profile
    def stochastic(self, state):
        return self.stochastic_with_probability(state)[0]

    def stochastic_with_probability(self, state):
        p = self.predict_costs(state)
        if len(state.actions) != self.n_actions:
            self.disallow.zero_()
            for i in range(self.n_actions):
                if i not in state.actions:
                    self.disallow[i] = 1e10
            p += Varng(self.disallow)
        probs = F.softmax(-p, dim=0)
        #print -probs.data.dot(torch.log(probs.data))
        return util.sample_from_probs(probs)

#    @profile
    def predict_costs(self, state):
        "Predict costs using the csoaa model accounting for `state.actions`"
        if self.features is None:
            #assert False
            feats = Varng(self.unit.zero_())
        else:
            feats = self.features(state)   # 77% time

        res = self.mapping(feats)

        return res.squeeze()
        #return self._lts_csoaa_predict(feats)  # 33% time

#    @profile
    def greedy(self, state, pred_costs=None):
        if pred_costs is None:
            pred_costs = self.predict_costs(state).data
        if isinstance(pred_costs, Var):
            pred_costs = pred_costs.data
        if len(state.actions) == self.n_actions:
            return pred_costs.min(0)[1].numpy()[0]
        best = None
        for a in state.actions: # 30% of time
            if best is None or pred_costs[a] < pred_costs[best]:  #62% of time
                best = a
        return best

#    @profile
    def forward_partial_complete(self, pred_costs, truth, actions):
        if isinstance(truth, np.int64):
            truth =[int(truth)]
        if isinstance(truth, int):
            truth = [truth]
        if isinstance(truth, list):
            truth0 = truth
            truth = torch.ones(self.n_actions)
            for k in truth0:
                truth[k] = 0.
        if not isinstance(truth, torch.FloatTensor):
            raise ValueError('lts_objective got truth of invalid type (%s)'
                             'expecting int, list[int] or torch.FloatTensor'
                             % type(truth))
        if len(actions) == self.n_actions:
            return self.distance(pred_costs, Var(truth, requires_grad=False))
        else:
            obj = 0.
            for a in actions:
                truth_a = self.unit.zero_() + truth[a]
                obj += self.distance(pred_costs[a], Var(truth_a, requires_grad=False))
            return obj


#    @profile
    def forward(self, state, truth=None):
        if truth is None:
            return self.greedy(state)
        # TODO: It would be better (more general) take a cost vector as input.
        # TODO: don't ignore limit_actions (timv: @hal3 is this fixed now that we call predict_costs?)

        # truth must be one of:
        #  - None: ignored
        #  - an int specifying the single true output (which gets cost zero, rest are cost one)
        #  - a list of ints specifying multiple true outputs (ala above)
        #  - a 1d torch tensor specifying the exact costs of every action
        costs = self.predict_costs(state)   # 47% of time (train)
#        print 'truth %s\tpred %s\tactions %s\tcosts %s' % \
#            (truth, self.greedy(state, limit_actions), limit_actions, list(pred_costs.data[0]))
        return self.forward_partial_complete(costs, truth, state.actions)  # 53% of time (train)
