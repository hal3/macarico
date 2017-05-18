from __future__ import division

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable

import macarico

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

    def __str__(self):
        return ' '.join(map(str, self.labels))


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
        self.actions = set(range(n_labels))
        super(SequenceLabeling, self).__init__(n_labels)

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
        return HammingLoss(self.example.labels).reference()

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

class HammingLossReference(macarico.Reference):
    def __init__(self, labels):
        self.labels = labels

    def __call__(self, state):
        return self.labels[state.n]

    def set_min_costs_to_go(self, state, cost_vector):
        cost_vector *= 0
        cost_vector += 1
        cost_vector[self.labels[state.n]] = 0.

class HammingLoss(object):
    def __init__(self, labels):
        self.labels = labels

    def __call__(self, env):
        assert len(env.output) == env.N, 'can only evaluate loss at final state'
        return sum(y != p for p,y in zip(env.output, self.labels)) #/ len(self.labels)

    def reference(self):
        return HammingLossReference(self.labels)

