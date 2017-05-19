from __future__ import division

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

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

    def __init__(self, example, n_labels, EOS=0, c_sub=1., c_ins=1., c_del=1.):
        self.N = len(example.tokens)
        self.T = self.N*2
        self.t = None
        self.tokens = example.tokens
        self.example = example
        self.EOS = EOS
        self.n = None
        self.output = []
        self.actions = set(range(n_labels))
        self.ref = EditDistanceReference(example.labels, c_sub, c_ins, c_del)
        super(Seq2Seq, self).__init__(n_labels)

    def rewind(self):
        self.t = None
        self.n = None
        self.output = []
        
    def run_episode(self, policy):
        self.output = []
        for self.t in xrange(self.T):
            a = policy(self)
            if a == self.EOS:
                break
            self.output.append(a)
        return self.output

    def loss(self):
        return self.ref.loss(self)

    def reference(self):
        return self.ref

    
class FrontBackAttention(macarico.Attention):
    """
    Attend to front and end of input string; if run with a BiLStM
    (eg), this should be sufficient to capture whatever you want.
    """
    arity = 2
    def __init__(self, field='tokens_rnn'):
        super(FrontBackAttention, self).__init__(field)

    def __call__(self, state):
        return [0, state.N-1]

class SoftmaxAttention(macarico.Attention, nn.Module):
    arity = None  # attention everywhere!
    
    def __init__(self, input_features, d_state, name_state='h'):
        nn.Module.__init__(self)

        self.input_features = input_features
        self.d_state = d_state
        self.name_state = name_state
        self.d_input = input_features.dim + d_state
        self.mapping = nn.Linear(self.d_input, 1)
        self.softmax = nn.Softmax()

        macarico.Attention.__init__(self, input_features.output_field)

    def __call__(self, state):
        N = state.N
        fixed_inputs = self.input_features(state)
        hidden_state = getattr(state, self.name_state)[state.t-1] if state.t > 0 else \
                       getattr(state, self.name_state + '0')
        #print fixed_inputs
        output = torch.cat([fixed_inputs.squeeze(1), hidden_state.repeat(3,1)], 1)
        return self.softmax(self.mapping(output)).view(1,-1)


class EditDistanceReference(macarico.Reference):
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

    def loss(self, env):
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

    def __call__(self, state):
        self.advance_to(state.output)
        A = set()
        for n in xrange(self.N):
            if self.prev_row[n] == self.prev_row_min:
                A.add( self.y[n] )
        return random.choice(list(A))

    def set_min_costs_to_go(self, state, cost_vector):
        self.advance_to(state.output)
        cost_vector *= 0
        cost_vector += self.c_del # you can always just delete something
        for n in xrange(self.N):
            l = self.y[n]
            cost_vector[l] = min(cost_vector[l], self.prev_row[n])
