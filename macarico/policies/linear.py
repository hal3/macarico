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
        return self.stochastic(state).view(1)[0]   # get an integer instead of pytorch.variable

    def stochastic(self, state):
        p = self.predict_costs(state)
        if len(state.actions) != self.n_actions:
            for i in range(self.n_actions):
                if i not in state.actions:
                    p[0,i] = 1e10
        return F.softmax(-p).multinomial()  # sample from softmin (= softmax on -costs)

    #@profile
    def predict_costs(self, state):
        "Predict costs using the csoaa model accounting for `state.actions`"
        feats = self.features(state)   # 77% time
        return self._lts_csoaa_predict(feats)  # 33% time

    #@profile
    def greedy(self, state, pred_costs=None):
        if pred_costs is None:
            pred_costs = self.predict_costs(state).data.numpy()  # 8% of time (train)
        if isinstance(pred_costs, Variable):
            pred_costs = pred_costs.data.numpy()
        if len(state.actions) == self.n_actions:
            return pred_costs.argmin()
        best = None
        for a in state.actions: # 30% of time
            if best is None or pred_costs[0,a] < pred_costs[0,best]:  #62% of time
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
                truth[0,k] = 0.
        if not isinstance(truth, torch.FloatTensor):
            raise ValueError('lts_objective got truth of invalid type (%s)'
                             'expecting int, list[int] or torch.FloatTensor'
                             % type(truth))
#        truth = Variable(truth, requires_grad=False)
        #print 'pred=%s\ntruth=%s\n' % (pred_costs, truth)
        truth = truth.view(-1, self.n_actions)
        if True:  # True = Fast version (marginally faster for dependency parser, way faster for seq2seq with large output spaces)
            if len(actions) != self.n_actions: # need to erase some
                a_vec = torch.zeros(truth.size())
                for a in actions: a_vec[0,a] = 1
                truth = (a_vec) * truth + (1 - a_vec) * pred_costs.data
            return self._lts_loss_fn(pred_costs, Variable(truth, requires_grad=False))
        else:
            obj = 0.
            for a in actions:
                obj += 0.5 * (pred_costs[0,a] - truth[0,a]) ** 2   # 89% of time (train)
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
