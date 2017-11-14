from __future__ import division, generators, print_function

import sys
import random
import macarico

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

class EditDistance(macarico.Loss):
    def __init__(self):
        super(EditDistance, self).__init__('edit')
        self.edr = EditDistanceReference()

    def evaluate(self, ex, env):
        edr = self.edr
        edr.y = ex.labels
        edr.N = len(edr.y)
        edr.reset()
        if edr.on_gold_path(env._trajectory):
            return (edr.N-1-len(env._trajectory)) * edr.c_ins
        
        edr.advance_to(env._trajectory)
        best_cost = None
        for n in range(edr.N):
            # if we aligned the most recent item to position n,
            # we would pay row[n] up to that point, and then
            # an additional (N-1)-n for inserting the rest
            this_cost = (edr.N-1-n) * edr.c_ins + edr.prev_row[n]
            if best_cost is None or this_cost < best_cost:
                best_cost = this_cost
        return best_cost
    
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
