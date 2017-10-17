from __future__ import division
import sys
import numpy as np
import random
import macarico

from macarico.tasks import seq2seq

class Example(object):
    def __init__(self, src, tgt, n_labels):
        self.tokens = src
        self.labels = tgt
        self.n_labels = n_labels

    def mk_env(self):
        assert self.labels[-1] == 0, 'target string must end in "0"==EOS'
        assert 0 not in self.tokens, 'source string must not contain "0"==EOS'
        return StringEdit(self)

    def __str__(self):
        return ' '.join(map(str, self.labels))

class StringEdit(macarico.Env):
    def __init__(self, ex):
        self.N = len(ex.tokens)
        self.M = len(ex.labels)
        self.n_labels = ex.n_labels
        self.T = self.N*2
        self.t = None # position in output
        self.n = None # position in tokens
        self.example = ex
        self.tokens = ex.tokens
        # actions:
        #   0..(n_labels-1) = insert a
        #   n_labels = copy
        #   n_labels+1 = delete
        self.output = []

        self.COPY = self.n_labels+0
        self.DELETE = self.n_labels+1
        
        self.n_actions = self.DELETE+1
        self.actions = set(range(self.n_actions))
        
        super(StringEdit, self).__init__(self.n_actions)

    def rewind(self):
        self.t = None
        self.n = None
        self.output = []
        
    def run_episode(self, policy):
        self.output = []
        self.n = 0
        for self.t in xrange(self.T):
            a = policy(self)
            if a < self.n_labels: # insert
                if a == 0: # EOS
                    break
                self.output.append(a)
            elif a == self.COPY:
                self.output.append(self.example.tokens[self.n])
                self.n += 1
            elif a == self.DELETE:
                self.n += 1

            # make sure n stays in bounds
            self.n = max(0, min(self.n, self.N-1))
        return self.output
#
#class StringEditLoss(macarico.Loss):
#    def __init__(self):
#        super(EditDistance, self).__init__('edit')
# #       self.ser = StringEditReference()
##
#    def evaluate(self, ex, env):
        

class StringEditReference(macarico.Reference):
    def __init__(self, n_labels, c_sub=1, c_ins=1, c_del=1, faux_cost=0.1):
        self.edr = seq2seq.EditDistanceReference(c_sub, c_ins, c_del)
        self.faux_cost = faux_cost
        self.my_cost_vector = np.zeros(n_labels+2)
        self.n_labels = n_labels
        
    def reset(self):
        self.edr.reset()

    def set_min_costs_to_go(self, state, cost_vector):
        if False and (state.t == 0 or True):
            if state.t == 0: print ''
            print state.example.tokens, state.example.labels, state.output
        #print self.n_labels, cost_vector.shape, state.COPY, state.DELETE, state.n_labels
        self.edr.set_min_costs_to_go(state, cost_vector)
        #print self.n_labels, state.COPY, state.DELETE, cost_vector.shape
        # the cost of COPY is the cost of action src[i]
        cv_min = cost_vector.min()
        w = state.tokens[state.n]
        cost_vector[state.COPY] = cost_vector[w]
        if cost_vector[w] <= cv_min:
            cost_vector[w] += self.faux_cost
        # the cost of DELETE is faux_delete_cost
        if state.n+1 < state.N:
            cost_vector[state.DELETE] = min(self.faux_cost,
                                            cost_vector[state.tokens[state.n+1]])

    def __call__(self, state):
        self.my_cost_vector *= 0
        self.set_min_costs_to_go(state, self.my_cost_vector)
        i_min = None
        min_count = 1
        for i in xrange(len(self.my_cost_vector)):
            if i not in state.actions: continue
            if i_min is None or self.my_cost_vector[i] < self.my_cost_vector[i_min]:
                i_min = i
                min_count = 1
            elif self.my_cost_vector[i] == self.my_cost_vector[i_min]:
                min_count += 1
                if np.random.random() < 1 / min_count:
                    i_min = i
        return i_min
        
class StringEditLoss(seq2seq.EditDistance):
    pass # no difference
