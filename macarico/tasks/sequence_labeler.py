from __future__ import division

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import macarico


zeros = lambda d: Variable(torch.zeros(1,d))
onehot = lambda i: Variable(torch.LongTensor([i]))


class Example(object):
    """
    >>> e = Example('abcdef', 'ABCDEF', 7)
    >>> env = e.mk_env()
    >>> env.run_episode(env.reference())
    ['A', 'B', 'C', 'D', 'E', 'F']
    >>> env.loss()
    0.0
    >>> env = e.mk_env()
    >>> env.run_episode(lambda s: s.tokens[s.n].upper() if s.n % 2 else '_')
    ['_', 'B', '_', 'D', '_', 'F']
    >>> env.loss()
    0.5
    """

    def __init__(self, tokens, labels, n_labels):
        self.tokens = tokens
        self.labels = labels
        self.n_labels = n_labels

    def mk_env(self):
        return SequenceLabeling(self, self.n_labels)


class SequenceLabeling(macarico.Env):
    """Basic sequence labeling environment (input and output sequences have the same
    length). Loss is evaluated with Hamming distance, which has an optimal
    reference policy.

    """

    def __init__(self, example, n_labels):
        self.example = example
        self.N = len(example.tokens)
        self.T = self.N
        self.n = None
        self.t = None
        self.prev_action = None          # previous action
        self.output = []
        self.tokens = example.tokens
        self.n_actions = n_labels
        self.actions = np.array(range(self.n_actions))

    def rewind(self):
        self.n = None
        self.t = None
        self.prev_action = None          # previous action
        self.output = []

    def run_episode(self, policy):
        self.output = []
        for self.n in xrange(self.N):
            self.t = self.n
            a = policy(self)
            self.output.append(a)
        return self.output

    def loss(self):
        return HammingLoss(self.example.labels)(self)

    def reference(self):
        return HammingLoss(self.example.labels).reference


class SeqFoci(object):
    """Attend to the current token's *input* embedding.

    TODO: We should be able to attend to the *output* embeddings too, i.e.,
    embedding of the previous actions and hidden states.

    TODO: Will need to cover boundary token embeddings in some reasonable way.

    """
    arity = 1
    def __init__(self, field='tokens_rnn'):
        self.field = field
    def __call__(self, state):
        return [state.n]


class RevSeqFoci(object):
    arity = 1
    def __init__(self, field='tokens_rnn'):
        self.field = field
    def __call__(self, state):
        return [state.N-state.n-1]


class HammingLoss(object):

    def __init__(self, labels):
        self.labels = labels

    def __call__(self, env):
        assert len(env.output) == env.N, 'can only evaluate loss at final state'
        return sum(y != p for p,y in zip(env.output, self.labels)) / len(self.labels)

    def reference(self, state):
        return self.labels[state.n]


class TransitionRNN(macarico.Features, nn.Module):

    def __init__(self,
                 sub_features,
                 foci,
                 n_actions,
                 d_actemb = 5,
                 d_hid = 50,
                ):
        nn.Module.__init__(self)

        # model is:
        #   h[-1] = zero
        #   for n in xrange(N):
        #     ae   = embed_action(y[n-1]) or zero if n=0
        #     h[n] = combine([f for f in foci], ae, h[n-1])
        #     y[n] = act(h[n])
        # we need to know:
        #   d_hid     - hidden state
        #   d_actemb  - action embeddings

        self.d_actemb = d_actemb
        self.d_hid = d_hid
        self.sub_features = {}
        for f in sub_features:
            field = f.output_field
            if field in self.sub_features:
                raise ValueError('multiple feature functions using same output field "%s"' % field)
            self.sub_features[field] = f

        # focus model; compute dimensionality
        self.foci = foci
        input_dim = self.d_actemb + self.d_hid
        for focus in self.foci:
            if focus.field not in self.sub_features:
                raise ValueError('focus asking for field "%s" but this does not exist in the constructed sub-features' % focus.field)
            input_dim += focus.arity * self.sub_features[focus.field].dim

        # nnet models
        self.embed_a = nn.Embedding(n_actions, self.d_actemb)
        self.combine = nn.Linear(input_dim, self.d_hid)

        macarico.Features.__init__(self, self.d_hid)

    def forward(self, state):
        t = state.t

        if not hasattr(state, 'h') or state.h is None:
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
        #inputs = [state.r[i] if i is not None else zeros(self.d_rnn*2) for i in self.foci(state)] + [ae, prev_h]
        inputs = [ae, prev_h]
        for focus in self.foci:
            idx = focus(state)
            assert len(idx) == focus.arity, \
                'focus %s is lying about its arity (claims %d, got %s)' % \
                (focus, focus.arity, idx)
            feats = self.sub_features[focus.field](state)
            for i in idx:
                if i is None:
                    # TODO: None as out of bounds
                    inputs.append(zeros(self.sub_features[focus.field].dim))
                else:
                    inputs.append(feats[i])

        state.h[t] = F.tanh(self.combine(torch.cat(inputs, 1)))

        return state.h[t]


class RNNFeatures(macarico.Features, nn.Module):

    def __init__(self,
                 n_types,
                 input_field = 'tokens',
                 output_field = 'tokens_rnn',
                 d_emb = 50,
                 d_rnn = 50,
                 bidirectional = True,
                 n_layers = 1,
                 rnn_type = nn.LSTM):
        # model is:
        #   embed words using standard embeddings, e[n]
        #   run biLSTM backwards over e[n], get r[n] = biLSTM state
        # we need to know dimensionality for:
        #   d_emb     - word embedding e[]
        #   d_rnn     - dimensionality
        #   n_layers  - how many layers of RNN
        #   bidirectional - is the RNN bidirectional?
        #   rnn_type - RNN/GRU/LSTM?

        nn.Module.__init__(self)

        self.input_field = input_field
        self.output_field = output_field
        self.d_emb = d_emb
        self.d_rnn = d_rnn
        self.embed_w = nn.Embedding(n_types, self.d_emb)
        self.rnn = rnn_type(self.d_emb,
                            self.d_rnn,
                            num_layers = n_layers,
                            bidirectional = bidirectional)

        macarico.Features.__init__(self, d_rnn * (2 if bidirectional else 1))

    def forward(self, state):
        if not hasattr(state, self.output_field) or \
           getattr(state, self.output_field) is None:
            # run a BiLSTM over input on the first step.
            my_input = getattr(state, self.input_field)
            e = self.embed_w(Variable(torch.LongTensor(my_input)))
            [res, _] = self.rnn(e.view(state.N,1,-1))
            setattr(state, self.output_field, res)

        return getattr(state, self.output_field)
