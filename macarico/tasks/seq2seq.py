from __future__ import division

import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import macarico

class Seq2Seq(macarico.Env):
    def __init__(self, tokens, EOS=0):
        self.N = len(tokens)
        self.T = self.N*2
        self.tokens = tokens
        self.EOS = EOS
        self.prev_action = None
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
            self.prev_action = a
            self.output.append(a)
        return self.output
    
    def loss_function(self, truth):
        return EditDistance(self, truth)

    def loss(self, truth):
        return self.loss_function(truth)()

class Seq2SeqFoci(object):
    """
    Attend to front and end of input string; if run with a BiLStM
    (eg), this should be sufficient to capture whatever you want.
    """
    arity = 2
    def __call__(self, state):
        return [0, state.N-1]
    
class EditDistance(object):
    def __init__(self, env, y, c_sub=1, c_ins=1, c_del=1):
        self.env = env
        self.y = y
        self.N = len(y)
        self.c_sub = c_sub
        self.c_ins = c_ins
        self.c_del = c_del
        self.reset()

    def reset(self):
        self.prev_row = [0] * self.N
        self.cur_row  = [0] * self.N
        for n in range(self.N):
            self.prev_row[n] = self.c_del * n
        self.prev_row_min = 0
        self.cur = []

    def __call__(self):
        env = self.env
        self.advance_to(env.output)
        best_cost = None
        for n in range(self.N):
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
        for n in range(1, self.N):
            self.cur_row[n] = min(min(self.prev_row[n] + self.c_ins,
                                      self.cur_row[n-1] + self.c_del),
                                  self.prev_row[n-1] + (0 if self.y[n-1] == p else self.c_sub))
            self.prev_row_min = min(self.prev_row_min, self.cur_row[n])
        tmp = self.cur_row
        self.cur_row = self.prev_row
        self.prev_row = tmp
        
    def reference(self, state, limit_actions=None):
        self.advance_to(state.output)
        A = set()
        for n in range(self.N):
            if self.prev_row[n] == self.prev_row_min:
                A.add( self.y[n] )
        return list(A)
    
