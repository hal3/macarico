from __future__ import division

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import macarico


zeros = lambda d: Variable(torch.zeros(1,d))
onehot = lambda i: Variable(torch.LongTensor([i]))


class SequenceLabeling(macarico.Env):
    """Basic sequence labeling environment (input and output sequences have the same
    length). Loss is evaluated with Hamming distance, which has an optimal
    reference policy.

    >>> e = SequenceLabeling('abcdefg')
    >>> target = list('ABCDEFG')
    >>> l = e.loss_function(target)
    >>> assert e.run_episode(l.reference) == target
    >>> l()
    0
    >>> e.output = 'ABC___G'
    >>> l()
    3

    """

    def __init__(self, tokens):
        self.N = len(tokens)
        self.T = self.N
        self.n = None
        self.tokens = tokens
        self.prev_action = None          # previous action
        self.output = []           # current output buffer

    def run_episode(self, policy):
        self.output = []
        for self.n in xrange(self.N):
            a = policy(self)
            self.prev_action = a
            self.output.append(a)
        return self.output

    def loss_function(self, true_labels):
        return HammingLoss(self, true_labels)

    def loss(self, true_labels):
        return self.loss_function(true_labels)()


class SeqFoci:
    """Attend to the current token's *input* embedding.

    TODO: We should be able to attend to the *output* embeddings too, i.e.,
    embedding of the previous actions and hidden states.

    TODO: Will need to cover boundary token embeddings in some reasonable way.

    """

    arity = 1

    def __call__(self, state):
        return [state.n]


class HammingLoss(object):

    def __init__(self, env, labels):
        self.env = env
        self.labels = labels
        assert len(labels) == env.N

    def __call__(self):
        env = self.env
        assert len(env.output) == env.N, 'can only evaluate loss at final state'
        return sum(y != p for p,y in zip(env.output, self.labels))

    def reference(self, state, limit_actions=None):
        return self.labels[state.n]


class BiLSTMFeatures(macarico.Features, nn.Module):

    def __init__(self, foci, n_words, n_labels, **kwargs):
        nn.Module.__init__(self)
        # model is:
        #   embed words using standard embeddings, e[n]
        #   run biLSTM backwards over e[n], get r[n] = biLSTM state
        #   h[-1] = zero
        #   for n in range(N):
        #     ae   = embed_action(y[n-1]) or zero if n=0
        #     h[n] = combine([r[i] for i in foci], ae, h[n-1])
        #     y[n] = act(h[n])
        # we need to know dimensionality for:
        #   d_emb     - word embedding e[]
        #   d_rnn     - RNN state r[]
        #   d_actemb  - action embeddings p[]
        #   d_hid     - hidden state
        #   n_layers  - how many layers of RNN
        #   n_foci    - how many tokens in input are combined for a prediction
        self.d_emb    = kwargs.get('d_emb',    50)
        self.d_rnn    = kwargs.get('d_rnn',    self.d_emb)
        self.d_actemb = kwargs.get('d_actemb', 5)
        self.d_hid    = kwargs.get('d_hid',    self.d_emb)
        self.n_layers = kwargs.get('n_layers', 1)

        # Focus model.
        self.foci = foci

        # TODO: figure out how to get dropout to work. There is a problem
        # between train and test in how dropout works. (We already need a
        # train/test time flag for reinforce to go from stoch to greedy. Also to
        # disable reference interpolation for SEARN).

        # set up simple sequence labeling model, which runs a biRNN
        # over the input, and then predicts left-to-right
        self.embed_w = nn.Embedding(n_words, self.d_emb)
        self.rnn = nn.RNN(self.d_emb, self.d_rnn, self.n_layers,
                          bidirectional=True) #dropout=kwargs.get('dropout', 0.5))
        self.embed_a = nn.Embedding(n_labels, self.d_actemb)
        self.combine = nn.Linear(self.d_rnn*2*foci.arity + self.d_actemb + self.d_hid,
                                 self.d_hid)

        macarico.Features.__init__(self, self.d_rnn)

    def forward(self, state):
        if state.prev_action is None:
            # run a BiLSTM over input on the first step.
            e = self.embed_w(Variable(torch.LongTensor(state.tokens)))
            [state.r, _] = self.rnn(e.view(state.N,1,-1))
            prev_h = zeros(self.d_hid)
            ae = zeros(self.d_actemb)
        else:
            prev_h = state.h
            # embed the previous action (if it exists)
            ae = self.embed_a(onehot(state.prev_action))

        # Combine input embedding, prev hidden state, and prev action embedding
        inputs  = [state.r[i] if i is not None else zeros(self.d_rnn*2) for i in self.foci(state)] + [ae, prev_h]
        state.h = F.tanh(self.combine(torch.cat(inputs, 1)))

        return state.h
