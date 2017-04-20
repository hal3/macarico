import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from macarico.annealing import ExponentialAnnealing


class Policy(object):
    def __call__(self, state):
        raise NotImplementedError('abstract')


class HammingReference(Policy):
    def __init__(self, labels):
        self.labels = labels
    def __call__(self, state):
        return self.labels[state.t]


class Env(object):
    def run_episode(self, policy):
        pass


class SequenceLabeling(Env):
    def __init__(self, tokens):
        self.T = len(tokens)
        self.tokens = tokens
        self.t = None          # current position
        self.output = []       # current output buffer t==len(output)

    def run_episode(self, policy):
        self.output = []
        for self.t in xrange(self.T):
            self.output.append(policy(self))
        return self.output


class Features(object):
    def __init__(self, dim):
        self.dim = dim
    def forward(self, state):
        raise NotImplementedError('abstract method not defined.')


class LearningAlg(object):
    def __call__(self, state):
        raise NotImplementedError('abstract method not defined.')


class DAgger(LearningAlg):

    def __init__(self, reference, policy, p_rollin_ref):
        self.n_passes = 0.
        self.p_rollin_ref = p_rollin_ref
        self.policy = policy
        self.reference = reference
        self.objective = 0.0

    def __call__(self, state):
        self.objective += self.policy.forward(state, self.reference(state))
        if self.p_rollin_ref():
            return self.reference(state)
        else:
            return self.policy.greedy(state)

    def update(self, _):
        self.objective.backward()


class Reinforce(LearningAlg):
    "REINFORCE with a scalar baseline function."

    def __init__(self, policy, baseline):
        self.trajectory = []
        self.baseline = baseline
        self.policy = policy
        super(Reinforce, self).__init__()

    def update(self, loss):
        b = self.baseline()
        for a in self.trajectory:
            a.reinforce(b - loss)
        self.baseline.update(loss)
        torch.autograd.backward(self.trajectory[:], [None]*len(self.trajectory))

    def __call__(self, state):
        action = self.policy.stochastic(state)
        # log actions (and values for actor critic) taken along current trajectory
        self.trajectory.append(action)
        return action.data[0,0]   # return an integer


class EWMA(object):
    "Exponentially weighted moving average."

    def __init__(self, rate, initial_value = 0.0):
        self.rate = rate
        self.value = initial_value

    def update(self, x):
        self.value += self.rate*(x - self.value)

    def __call__(self):
        return self.value


class BiLSTMFeatures(Features, nn.Module):

    def __init__(self, n_words, n_labels, **kwargs):
        nn.Module.__init__(self)
        # model is:
        #   embed words using standard embeddings, e[n]
        #   run biLSTM backwards over e[n], get r[n] = biLSTM state
        #   h[-1] = zero
        #   for n in range(N):
        #     ae   = embed_action(y[n-1]) or zero if n=0
        #     h[n] = combine(r[n], ae, h[n-1])
        #     y[n] = act(h[n])
        # we need to know dimensionality for:
        #   d_emb     - word embedding e[]
        #   d_rnn     - RNN state r[]
        #   d_actemb  - action embeddings p[]
        #   d_hid     - hidden state
        self.d_emb    = kwargs.get('d_emb',    50)
        self.d_rnn    = kwargs.get('d_rnn',    self.d_emb)
        self.d_actemb = kwargs.get('d_actemb', 5)
        self.d_hid    = kwargs.get('d_hid',    self.d_emb)
        self.n_layers = kwargs.get('n_layers', 1)

        # initialize the parent class; this needs to know the
        # branching factor of the task (in this case, the branching
        # factor is exactly the number of labels), the dimensionality
        # of the thing that will be used to make that prediction, and
        # the reference policy. we tell the search task to
        # automatically handle the reference policy for us. this
        # _only_ works when there is a one-to-one mapping between our
        # output and the sequence of actions we take; otherwise we
        # would have to handle the reference policy on our own.

        # set up simple sequence labeling model, which runs a biRNN
        # over the input, and then predicts left-to-right
        self.embed_w = nn.Embedding(n_words, self.d_emb)
        self.rnn = nn.RNN(self.d_emb, self.d_rnn, self.n_layers,
                          bidirectional=True) #dropout=kwargs.get('dropout', 0.5))
        self.embed_a = nn.Embedding(n_labels, self.d_actemb)
        self.combine = nn.Linear(self.d_rnn*2 + self.d_actemb + self.d_hid,
                                 self.d_hid)

        Features.__init__(self, self.d_rnn)

    def forward(self, state):
        # a few silly helper functions to make things cleaner
        zeros  = lambda d: Variable(torch.zeros(1,d))
        onehot = lambda i: Variable(torch.LongTensor([i]))

        T = state.T
        t = state.t
        if t == 0:
            # run a BiLSTM over input on the first step.
            e = self.embed_w(Variable(torch.LongTensor(state.tokens)))
            [state.r, _] = self.rnn(e.view(T,1,-1))
            prev_h = zeros(self.d_hid)
        else:
            prev_h = state.h

        # make predictions left-to-right
        if t == 0:
            # embed the previous action (if it exists)
            ae = zeros(self.d_actemb)
        else:
            #print t, state.output
            #assert isinstance(state.output, list), state.output
            y_prev = state.output[t-1]
            #assert isinstance(y_prev, int), y_prev
            ae = self.embed_a(onehot(y_prev))

        # combine hidden state appropriately
        state.h = F.tanh(self.combine(torch.cat([state.r[t], ae, prev_h], 1)))

        return state.h


class LinearPolicy(Policy, nn.Module):
    """
    Linear policy trained with the cost-sensitive one-against-all strategy.
    """

    def __init__(self, features, n_actions):
        nn.Module.__init__(self)
        # set up cost sensitive one-against-all
        # TODO make this generalizable
        self._lts_csoaa_predict = nn.Linear(features.dim, n_actions)
        self._lts_loss_fn = torch.nn.MSELoss(size_average=False) # only sum, don't average
        self.features = features

    def __call__(self, state):
        return self.sample(state)

    def sample(self, state):
        return self.stochastic(state).data[0,0]   # get an integer instead of pytorch.variable

    def stochastic(self, state):
        # predict costs using csoaa model
        pred_costs = self._lts_csoaa_predict(self.features(state))
        # return a soft-min sample (==softmax on negative costs)
        return F.softmax(-pred_costs).multinomial()

    def greedy(self, state):
        # predict costs using the csoaa model
        pred_costs = self._lts_csoaa_predict(self.features(state))
        # return the argmin cost
        return pred_costs.data.numpy().argmin()

    def forward(self, state, truth):
        # truth must be one of:
        #  - None: ignored
        #  - an int specifying the single true output (which gets cost zero, rest are cost one)
        #  - a list of ints specifying multiple true outputs (ala above)
        #  - a 1d torch tensor specifying the exact costs of every action
        if truth is None:
            return 0.

        pred_costs = self._lts_csoaa_predict(self.features(state))

        if isinstance(truth, int):
            truth = [truth]
        if isinstance(truth, list):
            truth0 = truth
            truth = torch.ones(pred_costs.size())
            for k in truth0:
                truth[0,k] = 0.
        if isinstance(truth, torch.FloatTensor):
            truth = Variable(truth, requires_grad=False)
            return self._lts_loss_fn(pred_costs, truth)

        raise ValueError('lts_objective got truth of invalid type (%s)'
                         'expecting int, list[int] or torch.FloatTensor'
                         % type(truth))


def test():

#    n_words = 5
#    n_labels = 3
    data = [
        ([0,1,2,3,4,3], [1,2,0,1,2,1])
    ]

    # Simple sequence reversal task
#    data = [(range(i,i+5), list(reversed(range(i,i+5)))) for i in range(50)]

    if 1:
        n = 5
        data = []
        for _ in range(50):
            x = [random.choice(range(5)) for _ in range(n)]
            y = list(reversed(x))
            data.append((x,y))

        random.shuffle(data)

        train = data[:len(data)//2]
        dev = data[len(data)//2:]
    else:
        dev = None
        train = data

    words = {x for X, _ in data for x in X}
    labels = {y for _, Y in data for y in Y}

    n_words = len(words)
    n_labels = len(labels)

    print 'n_words: %s, n_labels: %s' % (n_words, n_labels)

    policy = LinearPolicy(BiLSTMFeatures(n_words, n_labels), n_labels)

    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)

    def loss(want, got):
        return sum(y!=p for y,p in zip(want, got)) / len(want)

    def evaluate(data):
        errors = 0.0
        for words, labels in data:
            output = SequenceLabeling(words).run_episode(policy)
            errors += loss(labels, output)
        return errors / len(data)

    _p_rollin_ref = ExponentialAnnealing(0.99)

    baseline = EWMA(0.8)

    for epoch in range(500):
        for words,labels in train:
            env = SequenceLabeling(words)

            if 0:
                p_rollin_ref = lambda: random.random() <= _p_rollin_ref(epoch)
                learner = DAgger(HammingReference(labels), policy, p_rollin_ref)
            else:
                learner = Reinforce(policy, baseline)

            optimizer.zero_grad()
            output = env.run_episode(learner)
            learner.update(loss(labels, output))
            optimizer.step()

        if epoch % 1 == 0:
            if dev:
                print 'error rate: train %g, dev: %g' % (evaluate(train), evaluate(dev))
            else:
                print 'error rate: train %g' % evaluate(train)


if __name__ == '__main__':
    test()
