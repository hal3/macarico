from __future__ import division

import torch
import numpy as np
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

    def __init__(self, tokens, n_labels):
        self.N = len(tokens)
        self.T = self.N
        self.n = None
        self.t = None
        self.tokens = tokens
        self.prev_action = None          # previous action
        self.output = []
        self.n_labels = n_labels

    def run_episode(self, policy):
        self.output = []
        A = np.array(range(self.n_labels))
        for self.n in xrange(self.N):
            self.t = self.n
            a = policy(self, limit_actions=A)
            self.output.append(a)
        return self.output

    def loss_function(self, true_labels):
        return HammingLoss(self, true_labels)

    def loss(self, true_labels):
        return self.loss_function(true_labels)()


class SeqFoci(object):
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

    def __init__(self,
                 foci,
                 n_words,
                 n_labels,
                 d_emb = 50,
                 d_actemb = 5,
                 d_rnn = None,
                 d_hid = None,
                 bidirectional = True,
                 n_layers = 1,
                 rnn_type = nn.LSTM):
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

        nn.Module.__init__(self)
        self.d_emb = d_emb
        self.d_rnn = d_rnn or d_emb
        self.d_actemb = d_actemb
        self.d_hid = d_hid or d_emb

        # Focus model.
        self.foci = foci

        # set up simple sequence labeling model, which runs a biRNN
        # over the input, and then predicts left-to-right
        self.embed_w = nn.Embedding(n_words, self.d_emb)
        self.embed_a = nn.Embedding(n_labels, self.d_actemb)

        self.rnn = rnn_type(self.d_emb,
                            self.d_rnn,
                            num_layers = n_layers,
                            bidirectional = bidirectional)

        b = 2 if bidirectional else 1
        self.combine = nn.Linear(self.d_rnn * b * foci.arity
                                 + self.d_actemb + self.d_hid,  # ->
                                 self.d_hid)

        macarico.Features.__init__(self, self.d_rnn)

    def forward(self, state):
        t = state.t

        if t == 0:
            # run a BiLSTM over input on the first step.
            e = self.embed_w(Variable(torch.LongTensor(state.tokens)))
            [state.r, _] = self.rnn(e.view(state.N,1,-1))
            state.h = [None]*state.T
            prev_h = Variable(torch.zeros(1, self.d_hid))
            ae = zeros(self.d_actemb)
        else:

            if state.h[t] is not None:
                return state.h[t]

            prev_h = state.h[t-1].resize(1, self.d_hid)
            # embed the previous action (if it exists)
            ae = self.embed_a(onehot(state.output[t-1]))

        # Combine input embedding, prev hidden state, and prev action embedding
        inputs = [state.r[i] if i is not None else zeros(self.d_rnn*2) for i in self.foci(state)] + [ae, prev_h]

        state.h[t] = F.tanh(self.combine(torch.cat(inputs, 1)))

        return state.h[t]
