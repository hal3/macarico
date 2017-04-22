from __future__ import division

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import macarico


zeros  = lambda d: Variable(torch.zeros(1,d))
onehot = lambda i: Variable(torch.LongTensor([i]))


class SequenceLabeling(macarico.Env):

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

    def loss_function(self, true_labels):
        return HammingLoss(self, true_labels)

    def loss(self, true_labels):
        return self.loss_function(true_labels)()


class HammingLoss(object):

    def __init__(self, env, labels):
        self.env = env
        self.labels = labels
        assert len(labels) == env.T

    def __call__(self):
        env = self.env
        assert len(env.output) == env.T, 'can only evaluate loss at final state'
        return sum(y != p for p,y in zip(env.output, self.labels))

    def reference(self, state):
        return self.labels[state.t]


class BiLSTMFeatures(macarico.Features, nn.Module):

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

        # set up simple sequence labeling model, which runs a biRNN
        # over the input, and then predicts left-to-right
        self.embed_w = nn.Embedding(n_words, self.d_emb)
        self.rnn = nn.RNN(self.d_emb, self.d_rnn, self.n_layers,
                          bidirectional=True) #dropout=kwargs.get('dropout', 0.5))
        self.embed_a = nn.Embedding(n_labels, self.d_actemb)
        self.combine = nn.Linear(self.d_rnn*2 + self.d_actemb + self.d_hid,
                                 self.d_hid)

        macarico.Features.__init__(self, self.d_rnn)

    def forward(self, state):
        T = state.T
        t = state.t

        if t == 0:
            # run a BiLSTM over input on the first step.
            e = self.embed_w(Variable(torch.LongTensor(state.tokens)))
            [state.r, _] = self.rnn(e.view(T,1,-1))
            prev_h = zeros(self.d_hid)
            ae = zeros(self.d_actemb)
        else:
            prev_h = state.h
            # embed the previous action (if it exists)
            ae = self.embed_a(onehot(state.output[t-1]))

        # Combine input embedding, prev hidden state, and prev action embedding
        state.h = F.tanh(self.combine(torch.cat([state.r[t], ae, prev_h], 1)))

        return state.h
