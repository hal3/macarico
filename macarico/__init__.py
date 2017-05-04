import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

infinity = float('inf')

class Env(object):
    def run_episode(self, policy):
        pass

    def rewind(self):
        raise NotImplementedError('abstract')
    
class Policy(object):
    def __call__(self, state):
        raise NotImplementedError('abstract')


class Features(object):
    def __init__(self, dim):
        self.dim = dim
    def forward(self, state):
        raise NotImplementedError('abstract method not defined.')


class LearningAlg(object):
    def __call__(self, state):
        raise NotImplementedError('abstract method not defined.')
    
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
        c = self.predict_costs(state)
        return F.softmax(-c).multinomial()  # sample from softmin (= softmax on -costs)

    def predict_costs(self, state):
        "Predict costs using the csoaa model accounting for `state.actions`"
        p = self._lts_csoaa_predict(self.features(state))
        #c = infinity*p
        c = p*0 + 100000000   # XXX: infinity breaks torch's softmax
        assert c.size(0) == 1
        for a in state.actions:
            assert a < self.n_actions, 'state.actions includes invalid actions!'
            c[0,a] = p[0,a]
        return c

    def greedy(self, state):
        c = self.predict_costs(state).data.numpy()
        return int(c.argmin())

    def forward_partial_complete(self, pred_costs, truth):
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
        return self._lts_loss_fn(pred_costs, truth)

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
        return self.forward_partial_complete(c, truth)
