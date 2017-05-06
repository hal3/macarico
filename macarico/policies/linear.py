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
            for i in range(len(p[0])):
                if i not in state.actions:
                    p[0,i] = 1e10
        return F.softmax(-c).multinomial()  # sample from softmin (= softmax on -costs)

    def predict_costs(self, state):
        "Predict costs using the csoaa model accounting for `state.actions`"
        return self._lts_csoaa_predict(self.features(state))

    def greedy(self, state):
        p = self.predict_costs(state).data.numpy()
#        if len(p[0]) == len(state.actions):
#            return int(p.argmin())
        best = None
        for a in state.actions:
            if best is None or p[0,a] < p[0,best]:
                best = a
        return best

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
        truth = Variable(truth, requires_grad=False)
        obj = 0.
        #print 'pred=%s\ntruth=%s\n' % (pred_costs, truth)
        for a in actions:
            obj += 0.5 * (pred_costs[0,a] - truth[0,a]) ** 2
#        for i in range(len(c[0])):
#            if i not in state.actions:
#                c[0,i] = 1e10
        return obj
        #return self._lts_loss_fn(pred_costs, truth)

    def forward(self, state, truth):
        # TODO: It would be better (more general) take a cost vector as input.
        # TODO: don't ignore limit_actions (timv: @hal3 is this fixed now that we call predict_costs?)

        # truth must be one of:
        #  - None: ignored
        #  - an int specifying the single true output (which gets cost zero, rest are cost one)
        #  - a list of ints specifying multiple true outputs (ala above)
        #  - a 1d torch tensor specifying the exact costs of every action

        c = self.predict_costs(state)
#        print 'truth %s\tpred %s\tactions %s\tcosts %s' % \
#            (truth, self.greedy(state, limit_actions), limit_actions, list(pred_costs.data[0]))
        return self.forward_partial_complete(c, truth, state.actions)
