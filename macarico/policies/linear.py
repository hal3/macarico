from __future__ import division
import random
#import torch
#from torch import nn
#from torch.autograd import Variable
#from torch.nn import functional as F
#from torch.nn.parameter import Parameter
#import torch.nn.functional as F
import dynet as dy
import numpy as np

from macarico import Policy
from macarico import util

class LinearPolicy(Policy):
    """Linear policy

    Notes:

    This policy can be trained with
    - policy gradient via `policy.stochastic().reinforce(reward)`

    - Cost-sensitive one-against-all linear regression (CSOAA) via
      `policy.forward(state, truth)`

    """

    def __init__(self, dy_model, features, n_actions, loss_fn='squared', n_layers=1, hidden_dim=None):
#        nn.Module.__init__(self)
        self.dy_model = dy_model
        # set up cost sensitive one-against-all
        # TODO make this generalizable
        self.n_actions = n_actions
        dim = 1 if features is None else features.dim
        #self._lts_csoaa_predict = nn.Linear(dim, n_actions)
        #self._lts_loss_fn = torch.nn.MSELoss(size_average=False) # only sum, don't average

        if n_layers > 1 and hidden_dim is None:
            hidden_dim = dim
            assert hidden_dim > 1, \
                'LinearPolicy with n_layers>1 need hidden_dim>1 specified'

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.layers = []
        for layer in xrange(n_layers):
            in_dim = dim if layer == 0 else hidden_dim
            out_dim = n_actions if layer == n_layers-1 else hidden_dim

            w = dy_model.add_parameters((out_dim, in_dim))
            b = dy_model.add_parameters(out_dim)
            self.layers.append((w, b))

        #self._lts_csoaa_predict_w = dy_model.add_parameters((n_actions, dim))
        #self._lts_csoaa_predict_b = dy_model.add_parameters(n_actions)

        if   loss_fn == 'squared': self.distance = dy.squared_distance
        elif loss_fn == 'huber':   self.distance = dy.huber_distance
        else: assert False, ('unknown loss function %s' % loss_fn)

        self.features = features

    def __call__(self, state, deviate_to=None):
        return self.greedy(state, deviate_to=deviate_to)   # Run greedy!

    def sample(self, state):
        return self.stochastic(state)
        #return self.stochastic(state).view(1)[0]   # get an integer instead of pytorch.variable

#    @profile
    def stochastic(self, state, temperature=1):
        return self.stochastic_with_probability(state, temperature)[0]

    def stochastic_probability(self, state, temperature=1):
        p = self.predict_costs(state)
        if len(state.actions) != self.n_actions:
            disallow = np.zeros(self.n_actions)
            for i in range(self.n_actions):
                if i not in state.actions:
                    disallow[i] = 1e10
            p += dy.inputTensor(disallow)
        probs = dy.softmax(- p / temperature)
        return probs

    def stochastic_with_probability(self, state, temperature=1):
        probs = self.stochastic_probability(state, temperature)
        return util.sample_from_probs(probs)

#    @profile
    def predict_costs(self, state, deviate_to=None):
        "Predict costs using the csoaa model accounting for `state.actions`"
        if self.features is None:
            #assert False
            #feats = Variable(torch.zeros(1,1), requires_grad=False)
            feats = dy.parameter(self.dy_model.add_parameters(1))
            self.features = lambda _: feats
        else:
            feats = self.features(state)   # 77% time

        #predict_we = dy.parameter(self._lts_csoaa_predict_w)
        #predict_be = dy.parameter(self._lts_csoaa_predict_b)
        #res = dy.affine_transform([predict_be, predict_we, feats])
        res = feats
        for (w, b) in self.layers:
            w_ = dy.parameter(w)
            b_ = dy.parameter(b)
            res = dy.affine_transform([b_, w_, res])

        if deviate_to is not None:
            assert False
            #eta = -1
            #W = predict_we.npvalue()
            #K = W.shape[0]
            ##dev = eta * (W.sum(axis=0)/(K-1) - (1+1/(K-1))*W[deviate_to])
            #dev = 1.0 * W[deviate_to]
            #self.features.deviate_by(state, dev)

        return res
        #return self._lts_csoaa_predict(feats)  # 33% time

#    @profile
    def greedy(self, state, pred_costs=None, deviate_to=None):
        if pred_costs is None:
            #pred_costs = self.predict_costs(state).data.numpy()  # 8% of time (train)
            pred_costs = self.predict_costs(state, deviate_to=deviate_to)
        if isinstance(pred_costs, dy.Expression):
            pred_costs = pred_costs.npvalue()
        if len(state.actions) == self.n_actions:
            return pred_costs.argmin()
        best = None
        for a in state.actions: # 30% of time
            if best is None or pred_costs[a] < pred_costs[best]:  #62% of time
                best = a
        return best

#    @profile
    def forward_partial_complete(self, pred_costs, truth, actions):
        if isinstance(truth, int):
            truth = [truth]
        if isinstance(truth, list):
            truth0 = truth
            truth = np.ones(self.n_actions)
            for k in truth0:
                truth[k] = 0.
        #if not isinstance(truth, torch.FloatTensor):
        #    raise ValueError('lts_objective got truth of invalid type (%s)'
        #                     'expecting int, list[int] or torch.FloatTensor'
        #                     % type(truth))
#        truth = Variable(truth, requires_grad=False)
        #print 'pred=%s\ntruth=%s\n' % (pred_costs, truth)
        #truth = truth.view(-1, self.n_actions)
        if True:  # True = Fast version (marginally faster for dependency parser, way faster for seq2seq with large output spaces)
            if len(actions) == self.n_actions:
                return self.distance(pred_costs, dy.inputTensor(truth))
                #return dy.squared_distance(pred_costs, dy.inputTensor(truth))
                #return dy.huber_distance(pred_costs, dy.inputTensor(truth))
                #return self._lts_loss_fn(pred_costs, Variable(truth, requires_grad=False))
            else:
                obj = 0.
                for a in actions:
                    #v = (pred_costs[a] - truth[a])
                    #obj += 0.5 * v * v
                    obj += self.distance(pred_costs[a], dy.inputTensor([truth[a]]))
                    #obj += dy.huber_distance(pred_costs[a], dy.inputTensor([truth[a]]))
                return obj
#            if len(actions) != self.n_actions: # need to erase some
#                a_vec = torch.zeros(truth.size())
#                for a in actions: a_vec[a] = 1
#                truth = (a_vec) * truth + (1 - a_vec) * pred_costs.data
#                assert(False)
#            return self._lts_loss_fn(pred_costs, Variable(truth, requires_grad=False))
        else:
            obj = 0.
            for a in actions:
                obj += 0.5 * (pred_costs[a] - truth[a]) ** 2   # 89% of time (train)
            return obj
#        for i in range(len(c[0])):
#            if i not in state.actions:
#                c[i] = 1e10

#    @profile
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
