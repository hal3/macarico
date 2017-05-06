from __future__ import division

import numpy as np
import random
import macarico

class Example(object):
    def __init__(self, tokens, labels, n_labels):
        self.tokens = tokens
        self.labels = labels
        self.n_labels = n_labels

    def mk_env(self):
        return Seq2Seq(self, self.n_labels)

    def __str__(self):
        return ' '.join(map(str, self.labels))


class Seq2Seq(macarico.Env):

    def __init__(self, example, n_labels, EOS=0):
        self.N = len(example.tokens)
        self.T = self.N*2
        self.t = None
        self.tokens = example.tokens
        self.example = example
        self.EOS = EOS
        self.n = None
        self.output = []
        self.actions = set(range(n_labels))

    def rewind(self):
        self.t = None
        self.n = None
        self.output = []
        
    def run_episode(self, policy):
        self.output = []
        for self.t in xrange(self.T):
            a = policy(self)
            if isinstance(a, list):
                a = random.choice(a)
            if a == self.EOS:
                break
            self.output.append(a)
        return self.output

    def loss_function(self, truth):
        return EditDistance(self, truth)

    def loss(self):
        return EditDistance(self.example.labels)(self)

    def reference(self):
        return EditDistance(self.example.labels).reference


class Seq2SeqFoci(object):
    """
    Attend to front and end of input string; if run with a BiLStM
    (eg), this should be sufficient to capture whatever you want.
    """
    arity = 2
    def __init__(self, field='tokens_rnn'):
        self.field = field
    def __call__(self, state):
        return [0, state.N-1]


class EditDistance(object):
    def __init__(self, y, c_sub=1, c_ins=1, c_del=1):
        self.y = y
        self.N = len(y)
        self.prev_row_min = None
        self.cur_row = None
        self.prev_row = None
        self.c_sub = c_sub
        self.c_ins = c_ins
        self.c_del = c_del
        self.reset()

    def reset(self):
        self.prev_row = [0] * self.N
        self.cur_row  = [0] * self.N
        for n in xrange(self.N):
            self.prev_row[n] = self.c_del * n
        self.prev_row_min = 0
        self.cur = []

    def __call__(self, env):
        self.advance_to(env.output)
        best_cost = None
        for n in xrange(self.N):
            # if we aligned the most recent item to position n,
            # we would pay row[n] up to that point, and then
            # an additional (N-1)-n for inserting the rest
            this_cost = (self.N-1-n) * self.c_ins + self.prev_row[n]
            if best_cost is None or this_cost < best_cost:
                best_cost = this_cost
        return best_cost

    def advance_to(self, pred):
        if not self.cur_extends(pred):
            self.reset()
        for x in pred[len(self.cur):]:
            self.step(x)

    def cur_extends(self, pred):
        return len(self.cur) >= len(pred) and \
            all((p == c for p,c in zip(pred, self.cur)))

    def step(self, p):
        self.cur.append(p)
        self.cur_row[0] = self.prev_row[0] + self.c_ins
        self.prev_row_min = self.cur_row[0]
        for n in xrange(1, self.N):
            self.cur_row[n] = min(min(self.prev_row[n] + self.c_ins,
                                      self.cur_row[n-1] + self.c_del),
                                  self.prev_row[n-1] + (0 if self.y[n-1] == p else self.c_sub))
            self.prev_row_min = min(self.prev_row_min, self.cur_row[n])
        tmp = self.cur_row
        self.cur_row = self.prev_row
        self.prev_row = tmp

    def reference(self, state):
        self.advance_to(state.output)
        A = set()
        for n in xrange(self.N):
            if self.prev_row[n] == self.prev_row_min:
                A.add( self.y[n] )
        return list(A)
