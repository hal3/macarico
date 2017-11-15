from __future__ import division, generators, print_function

import sys
import random
import macarico
from collections import deque

class Example(object):
    def __init__(self, tokens, labels, n_labels):
        assert all((i != Seq2Seq.EOS for i in labels))
        self.tokens = tokens
        self.labels = labels + [Seq2Seq.EOS]
        self.n_labels = n_labels

    def mk_env(self):
        return Seq2Seq(self, self.n_labels)

    def __str__(self):
        return ' '.join(map(str, self.labels[:-1]))


class Seq2Seq(macarico.Env):
    LENGTH_FACTOR = 4
    EOS = 0
    def __init__(self, example, n_labels):
        macarico.Env.__init__(self, n_labels)
        
        self.N = len(example.tokens)
        self.tokens = example.tokens
        self.example = example
        self._trajectory = []
        self.actions = set(range(n_labels))
        super(Seq2Seq, self).__init__(n_labels)

    def horizon(self):
        return self.N * Seq2Seq.LENGTH_FACTOR
        
    def _rewind(self):
        self._trajectory = []
        
    def _run_episode(self, policy):
        self._trajectory = []
        for self.n in range(self.N * Seq2Seq.LENGTH_FACTOR):
            a = policy(self)
            if a == Seq2Seq.EOS:
                break
            self._trajectory.append(a)
        return self._trajectory

def levenshteinDistance(s1, s2): # from https://stackoverflow.com/questions/2460177/edit-distance-in-python
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]
    
class EditDistance(macarico.Loss):
    def __init__(self):
        super(EditDistance, self).__init__('edit')

    def evaluate(self, ex, env, importance=1.0):
        assert ex.labels[-1] == Seq2Seq.EOS
        return levenshteinDistance(ex.labels[:-1], env._trajectory)

class NgramFollower(macarico.Reference):
    def __init__(self):
        macarico.Reference.__init__(self)
        self.y = None
        self.ngrams = {}

    def maybe_reset(self, labels, p):        
        if self.y is not None and len(self.y) == len(labels) and all((self.y[i] == labels[i] for i in range(len(self.y)))) and \
           len(self.last_p) < len(p) and all((a==b for a,b in zip(self.last_p, p))):
            return # we're all set
        
        self.y = labels
        self.ngrams = {}
        for i in range(len(labels)):
            next_word = int(labels[i])
            for l in range(i):
                ngram = ' '.join(labels[i-l-1:i])
                if ngram not in self.ngrams:
                    self.ngrams[ngram] = deque()
                self.ngrams[ngram].append(next_word)
        self.last_p = []
        self.untouched = list(map(int, labels[:]))
                
    def __call__(self, state):
        labels = list(map(str, state.example.labels[:-1]))
        p = list(map(str, state._trajectory))
        self.maybe_reset(labels, p)

        for i in range(len(self.last_p), len(p)):
            # need to remove any ngrams that end at i
            next_word = int(p[i])
            for l in range(i):
                ngram = ' '.join(labels[i-l-1:i])
                if ngram in self.ngrams:
                    try:
                        self.ngrams[ngram].remove(next_word)
                        if len(self.ngrams[ngram]) == 0:
                            del self.ngrams[ngram]
                    except ValueError: # not in deque
                        pass
            try:
                self.untouched.remove(next_word)
            except ValueError:
                pass

        self.last_p = p
                    
        for i in range(1,len(p)+1):
            suffix = ' '.join(p[-i:])
            if suffix in self.ngrams:
                w = self.ngrams[suffix][0]
                return w

        if len(self.untouched) == 0:
            return Seq2Seq.EOS
        return self.untouched[0] # this is not optimal, but eh
        
    
class EditDistanceReference(macarico.Reference):
    def __init__(self, c_sub=1, c_ins=1, c_del=1):
        macarico.Reference.__init__(self)
        self.prev_row_min = None
        self.cur_row = None
        self.prev_row = None
        self.c_sub = min(c_sub, c_ins + c_del)  # doesn't make sense otherwise
        self.c_ins = c_ins
        self.c_del = c_del
        self.y = None

    def reset(self):
        #print('y=', self.y)
        self.prev_row = [0] * self.N
        self.cur_row  = [0] * self.N
        for n in range(self.N):
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
        for n in range(1, self.N):
            self.cur_row[n] = min(min(self.prev_row[n] + self.c_ins,
                                      self.cur_row[n-1] + self.c_del),
                                  self.prev_row[n-1] + (0 if self.y[n-1] == p else self.c_sub))
            self.prev_row_min = min(self.prev_row_min, self.cur_row[n])
        tmp = self.cur_row
        self.cur_row = self.prev_row
        self.prev_row = tmp

    def not_same(self, labels):
        return self.y is None or \
            len(self.y) != len(labels) or \
            any((self.y[i] != labels[i] for i in range(len(self.y))))
    
    def __call__(self, state):
        if self.not_same(state.example.labels):
            self.y = state.example.labels
            self.N = len(self.y)
            self.reset()
        
        if self.on_gold_path(state._trajectory):
            return self.y[len(state._trajectory)]
        
        self.advance_to(state._trajectory)
        A = set()
        for n in range(self.N):
            if self.prev_row[n] == self.prev_row_min:
                A.add( self.y[n] )
        # debugging
        #if self.on_gold_path(state._trajectory):
        #    A = list(A)
        #    assert len(A) == 1
        #    assert A[0] == self.y[len(state._trajectory)]
        return random.choice(list(A))

    def set_min_costs_to_go(self, state, cost_vector):
        if self.not_same(state.example.labels):
            self.y = state.example.labels
            self.N = len(self.y)
            self.reset()
        
        cost_vector *= 0
        cost_vector += self.c_del # you can always just delete something
        
        if self.on_gold_path(state._trajectory):
            n = len(state._trajectory)
            cost_vector[self.y[n]] = 0
            cost_vector[0] = (self.N-1-n) * self.c_ins
            #print(cost_vector)
            return

        #print('not gold path')
        #import ipdb; ipdb.set_trace()
        print('urgh not on gold path and I think this reference might be broken, please fix!', file=sys.stderr)
        self.advance_to(state._trajectory)
        finish_cost = self.c_sub * min(self.N-1, len(state._trajectory)) + \
                      self.c_ins * max(0, self.N-1 - len(state._trajectory)) + \
                      self.c_del * max(0, len(state._trajectory) - self.N+1)
        for n in range(self.N):
            l = self.y[n]
            cost_vector[l] = min(cost_vector[l], self.prev_row[n])
            finish_cost = min(finish_cost, (self.N-1-n) * self.c_ins + self.prev_row[n])
        cost_vector[0] = finish_cost
        #print(cost_vector)

def test_edr():
    edr = EditDistanceReference([1,2,3,4,0])
    costs = torch.zeros(10)
    state = Example(0,0,0)
    state._trajectory = []
    for x in edr.y:
        edr.set_min_costs_to_go(state, costs)
        print(state._trajectory, edr.prev_row, costs)
        state._trajectory.append(x)
