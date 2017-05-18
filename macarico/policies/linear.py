from __future__ import division
import random
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from macarico import Policy

class LinearPolicy(Policy, nn.Module):
    """Linear policy

    Notes:

    This policy can be trained with
    - policy gradient via `policy.stochastic().reinforce(reward)`

    - Cost-sensitive one-against-all linear regression (CSOAA) via
      `policy.forward(state, truth)`

    """

    def __init__(self, features, n_actions):
        nn.Module.__init__(self)
        # set up cost sensitive one-against-all
        # TODO make this generalizable
        self.n_actions = n_actions
        self._lts_csoaa_predict = nn.Linear(features.dim, n_actions)
        self._lts_loss_fn = torch.nn.MSELoss(size_average=False) # only sum, don't average
        self.features = features

    def __call__(self, state):
        return self.greedy(state)   # Run greedy!

    def sample(self, state):
        return self.stochastic(state).data[0,0]   # get an integer instead of pytorch.variable

    def stochastic(self, state):
        p = self.predict_costs(state)
        if len(p) != len(state.actions):
            for i in range(len(p)):
                if i not in state.actions:
                    p[i] = 1e10
        return F.softmax(-c).multinomial()  # sample from softmin (= softmax on -costs)

    #@profile
    def predict_costs(self, state):
        "Predict costs using the csoaa model accounting for `state.actions`"
        feats = self.features(state)   # 70% time
        return self._lts_csoaa_predict(feats)[0]  # 30% time

    #@profile
    def greedy(self, state, pred_costs=None):
        if pred_costs is None:
            pred_costs = self.predict_costs(state).data.numpy()  # 97% of time (train)
#        if len(p[0]) == len(state.actions):
#            return int(p.argmin())
        best = None
        for a in state.actions:
            if best is None or pred_costs[a] < pred_costs[best]:
                best = a
        return best

    #@profile
    def forward_partial_complete(self, pred_costs, truth, actions):
        if isinstance(truth, int):
            truth = [truth]
        if isinstance(truth, list):
            truth0 = truth
            truth = torch.ones(pred_costs.size())
            for k in truth0:
                truth[k] = 0.
        if not isinstance(truth, torch.FloatTensor):
            raise ValueError('lts_objective got truth of invalid type (%s)'
                             'expecting int, list[int] or torch.FloatTensor'
                             % type(truth))
#        truth = Variable(truth, requires_grad=False)
        #print 'pred=%s\ntruth=%s\n' % (pred_costs, truth)
        obj = 0.
        for a in actions:
            obj += 0.5 * (pred_costs[a] - truth[a]) ** 2   # 89% of time (train)
        return obj
#        for i in range(len(c[0])):
#            if i not in state.actions:
#                c[0,i] = 1e10

    #@profile
    def forward(self, state, truth):
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
