from __future__ import division

import numpy as np
#import torch
#from torch import nn
#from torch.autograd import Variable

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
        self.output = []
        self.actions = set(range(n_labels))
        super(Seq2Seq, self).__init__(n_labels)

    def rewind(self):
        self.t = None
        self.output = []
        
    def run_episode(self, policy):
        self.output = []
        for self.t in xrange(self.T):
            a = policy(self)
            if a == self.EOS:
                break
            self.output.append(a)
        return self.output

class EditDistance(macarico.Loss):
    def __init__(self):
        super(EditDistance, self).__init__('edit')
        self.edr = EditDistanceReference()

    def evaluate(self, ex, env):
        edr = self.edr
        edr.y = ex.labels
        edr.N = len(edr.y)
        edr.reset()
        if edr.on_gold_path(env.output):
            return (edr.N-1-len(env.output)) * edr.c_ins
        
        edr.advance_to(env.output)
        best_cost = None
        for n in xrange(edr.N):
            # if we aligned the most recent item to position n,
            # we would pay row[n] up to that point, and then
            # an additional (N-1)-n for inserting the rest
            this_cost = (edr.N-1-n) * edr.c_ins + edr.prev_row[n]
            if best_cost is None or this_cost < best_cost:
                best_cost = this_cost
        return best_cost
    
class EditDistanceReference(macarico.Reference):
    def __init__(self, c_sub=1, c_ins=1, c_del=1):
        self.prev_row_min = None
        self.cur_row = None
        self.prev_row = None
        self.c_sub = min(c_sub, c_ins + c_del)  # doesn't make sense otherwise
        self.c_ins = c_ins
        self.c_del = c_del
        self.y = None

    def reset(self):
        self.prev_row = [0] * self.N
        self.cur_row  = [0] * self.N
        for n in xrange(self.N):
            self.prev_row[n] = self.c_del * n
        self.prev_row_min = 0
        self.cur = []

    def on_gold_path(self, out):
        return len(out) <= self.N-1 and \
            all((a == b for a,b in zip(out, self.y)))
        
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
        if self.y != state.example.labels:
            self.y = state.example.labels
            self.N = len(self.y)
            self.reset()
        
        if self.on_gold_path(state.output):
            return self.y[len(state.output)]
        
        self.advance_to(state.output)
        A = set()
        for n in xrange(self.N):
            if self.prev_row[n] == self.prev_row_min:
                A.add( self.y[n] )
        # debugging
        #if self.on_gold_path(state.output):
        #    A = list(A)
        #    assert len(A) == 1
        #    assert A[0] == self.y[len(state.output)]
        return random.choice(list(A))

    def set_min_costs_to_go(self, state, cost_vector):
        if self.y != state.example.labels:
            self.y = state.example.labels
            self.N = len(self.y)
            self.reset()
        
        cost_vector *= 0
        cost_vector += self.c_del # you can always just delete something
        
        if self.on_gold_path(state.output):
            n = len(state.output)
            cost_vector[self.y[n]] = 0
            cost_vector[0] = (self.N-1-n) * self.c_ins
            return
        
        self.advance_to(state.output)
        finish_cost = self.c_sub * min(self.N-1, len(state.output)) + \
                      self.c_ins * max(0, self.N-1 - len(state.output)) + \
                      self.c_del * max(0, len(state.output) - self.N+1)
        for n in xrange(self.N):
            l = self.y[n]
            cost_vector[l] = min(cost_vector[l], self.prev_row[n])
            finish_cost = min(finish_cost, (self.N-1-n) * self.c_ins + self.prev_row[n])
        cost_vector[0] = finish_cost

def test_edr():
    edr = EditDistanceReference([1,2,3,4,0])
    costs = np.zeros(10)
    state = Example(0,0,0)
    state.output = []
    for x in edr.y:
        edr.set_min_costs_to_go(state, costs)
        print state.output, edr.prev_row, costs
        state.output.append(x)
