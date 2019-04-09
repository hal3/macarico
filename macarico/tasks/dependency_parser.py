from __future__ import division, generators, print_function
from collections import defaultdict
import random
import torch
import macarico
from macarico.data.types import DependencyTree

class DependencyParser(macarico.Env):
    """
    A greedy transition-based parser, based heavily on
    Matthew Honnibal's "500 lines" implementation:
      https://gist.github.com/syllog1sm/10343947
    As well as Daniel Pressel's implementation:
      https://github.com/dpressel/arcs-py/
    And finally Kai-Wei Chang's implementation in VW.
    This is an arc-hybrid dependency parser
    """

    SHIFT, RIGHT, LEFT, N_ACT = 0, 1, 2, 3

    def __init__(self, example):
        self.n_rels = example.n_rels
        
        T = 2 * example.N * (1 if self.n_rels == 0 else 2)
        macarico.Env.__init__(self, self.N_ACT + self.n_rels, T, example)

        self.example = example
        self.X = example.X
        self.tags = example.tags
        self.N = example.N

        self.gold_heads = {}
        self.gold_deps = defaultdict(lambda: [])
        if self.example.Y is not None:
            for dep in range(self.N):
                head, _ = self.example.Y[dep]
                self.gold_heads[dep] = head
                if head not in self.gold_deps:
                    self.gold_deps[head] = []
                self.gold_deps[head].append(dep)
        #print 'gold_heads =', self.gold_heads
        #print 'gold_deps  =', self.gold_deps

        self.root = self.N
        self.b = 0   # invariant: buf = [b, b+1, ..., N]
        self.stack = []
        
        self.a = None
        self.Yhat = DependencyTree(self.N, self.n_rels>0)  # +1 for ROOT at end
        self.actions = None
        self.is_rel = None       # used to indicate whether the action type is a label action or not.
        if self.n_rels > 0:
            self.valid_rels = set(range(self.N_ACT, self.N_ACT+self.n_rels))
            
    def _rewind(self):
        #print '\n-------------------'
        self.a = None
        self.stack = []
        self.b = 0
        self.Yhat = DependencyTree(self.N, self.n_rels > 0)
        self.actions = None

    def __str__(self):
        return 'stack = %s\nb     = %d\narcs  = %s' % (self.stack, self.b, self.Yhat)
            #print 'stack = %s\tbuf = %s' % (self.stack, self.b)
        
        
    def _run_episode(self, policy):
        # run shift/reduce parser
        while True: #not (len(self.stack) == 0 and len(self.buf) == 1):
#            assert self.buf == range(self.b, self.N+1), \
#                'b=%d buf=%s' % (self.b, self.buf)
            # len(self.buf) == self.N - self.b+1
            # so len(buf) > 0 <==> self.N+1 - self.b > 0 <==> self.b < self.N+1 <==> self.b <= self.N
            if len(self.stack) == 0 and self.b == self.N:
                break
            # get shift/reduce action
            self.is_rel = False
            self.actions = self.get_valid_transitions()
            #print 'actions = %s' % self.actions
            self.a = policy(self)
            #print('stack = %s\tbuf = %s\tactions = %s\ta = %d' % (self.stack, self.b, self.actions, self.a))
            #print self
            #print self.actions
            #print self.a
            #print
            #print 'i=%d\tstack=%s\tparse=%s\ta=%s' % (self.i, self.stack, self.parse, self.a),
            assert self.a in self.actions, \
                'policy %s returned an invalid transition "%s" (must be one of %s)!' % (type(policy), self.a, self.actions)

            # if we're doing labeled parsing, get relation
            rel = None
            if self.n_rels > 0 and self.a != self.SHIFT:
                self.is_rel = True
                self.actions = self.valid_rels
                rel = policy(self)
                assert rel is not None
                #if rel is None:   # timv: @hal3 why will this ever be None?
                #    rel = random.choice(self.valid_rels)
                rel -= self.N_ACT

            self.transition(self.a, rel)

        return self.Yhat

    def get_valid_transitions(self):
        actions = set()
        
        if self.b < self.N: # len(self.buf) > 1:
            actions.add(self.SHIFT)
            
        if len(self.stack) >= 2:
            actions.add(self.RIGHT)
            
        if len(self.stack) >= 1 and self.b <= self.N and self.stack[-1] != self.root:
            actions.add(self.LEFT)
            
        return actions

    def transition(self, a, rel=None):
        #print 'transition %d in %s' % (a, self.actions)
        if a == self.SHIFT: # == 0
            self.stack.append(self.b)
            #del self.buf[0]
            self.b += 1
        elif a == self.RIGHT: # == 1
            # add(head, child); child will NEVER get another head
            s0 = self.stack.pop()
            s1 = self.stack[-1]
            self.Yhat.add(s1, s0, rel)
        elif a == self.LEFT: # == 2
            # in KW's code, heads[stack[-1]] = idx
            # so stack[-1] = child, idx = head
            s0 = self.stack.pop()
            #b = self.buf[0]
            self.Yhat.add(self.b, s0, rel)
        else:
            assert False, 'transition got invalid move %d' % a


class AttachmentLossReference(macarico.Reference):
    def __call__(self, state):
        if state.is_rel:
            return random.choice(self.relation_reference(state))
        else:
            costs = torch.zeros(3) + 99999999
            for a in state.actions: costs[a] = 0
            self.transition_costs(state, costs)
            #print costs
            ref = None
            for a in state.actions:
                if ref is None or costs[a] < costs[ref]:
                    ref = a
            return ref
            
    def set_min_costs_to_go(self, state, cost_vector):
        cost_vector *= 0
        if state.is_rel:
            cost_vector += 1
            for a in self.relation_reference(state):
                cost_vector[a] = 0
        else:
            self.transition_costs(state, cost_vector)
        
    def transition_costs(self, state, costs):
        # SHIFT=0: then b=buf[0] will be put onto the stack, and won't
        # be able to get heads from {s1}+S and will not be able to get
        # deps from {s0,s1}+S
        if state.b <= state.N and len(state.stack) > 0:
            if state.b in state.gold_heads:  # no
                for j in state.stack[0:-1]:
                    if j == state.gold_heads[state.b]:
                        #print('SHIFT+=1 because b=', state.b,'<N and in', state.gold_heads, 'and j=', j, '==gold_heads[b]=', state.gold_heads[state.b])
                        costs[state.SHIFT] += 1
            for dep in state.gold_deps[state.b]: # deps[2] = [0, 1]
                if dep in state.stack: # stack = [0, 1], so YES
                    #print('SHIFT+=1 because b=', state.b,'<N and dep=', dep, 'in', state.gold_deps[state.b], 'in stack=', state.stack)
                    costs[state.SHIFT] += 1

        # RIGHT=1: adding arc (s1,s0) and popping s0 means s0 won't be
        # able to acquire heads or deps from B
        if len(state.stack) >= 2:
            s0 = state.stack[-1]
            for i in range(state.b, state.N+1):
                if (i in state.gold_heads and state.gold_heads[i] == s0) or \
                   state.gold_heads[s0] == i:
                    costs[state.RIGHT] += 1


        # LEFT=2: adding arc (b,s0) and popping s0 from stack means s0
        # won't be able to acquire heads from {s1}+B nor dependents
        # from B+b.
        if len(state.stack) >= 1:
            s0 = state.stack[-1]
            for i in range(state.b+1, state.N+1):
                if (i < state.N and state.gold_heads[i] == s0) or state.gold_heads[s0] == i:
                    costs[state.LEFT] += 1
            if state.b < state.N and state.gold_heads[state.b] == s0:
                costs[state.LEFT] += 1
            if len(state.stack) >= 2 and state.gold_heads[s0] == state.stack[-2]:
                costs[state.LEFT] += 1

    def relation_reference(self, state):
        a = state.a
        if a == state.RIGHT:
            # new edge is parse.add(state.stack[-2], state.stack.pop(), rel)
            head  = state.stack[-2]
            child = state.stack[-1]
        elif a == state.LEFT:
            # new edge is parse.add(state.i, state.stack.pop(), rel)
            head  = state.b
            child = state.stack[-1]
        else:
            assert False, 'relation_reference called with a=%s was neither LEFT nor RIGHT' % a

        if state.gold_heads[child] == head:
            return [state.example.Y[child][1] + state.N_ACT]
        else:
            return list(state.actions)

class DependencyAttention(macarico.Attention):
    arity = 2
    def __init__(self, features):
        macarico.Attention.__init__(self, features)
        self.oob = self.make_out_of_bounds()

    def _forward(self, state):
        x = self.features(state)
        b = state.b
        i = state.stack[-1] if len(state.stack) > 0 else -1 # for left & right
        #i2 = state.stack[-2] if len(state.stack) > 1 else None # for right
        #print(b,i,state.N,x.shape)
        return [x[0,b].unsqueeze(0) if b >= 0 and b < state.N else self.oob,
                x[0,i].unsqueeze(0) if i >= 0 and i < state.N else self.oob]


class AttachmentLoss(macarico.Loss):
    def __init__(self, scale_by_length=False):
        self.scale_by_length = scale_by_length
        super(AttachmentLoss, self).__init__('lal')

    def evaluate(self, example):
        loss = 0
        #print(example.Y, example.Yhat, type(example))
        for (pred_head, pred_rel), (true_head, true_rel) in zip(example.Yhat, example.Y):
            if pred_head != true_head or \
               (example.n_rels > 0 and pred_rel != true_rel):
                loss += 1
        if self.scale_by_length:
            loss /= len(example.X)
        return loss

class GlobalAttachmentLoss(macarico.Loss):
    def __init__(self):
        super(GlobalAttachmentLoss, self).__init__('glal', corpus_level=True)
        self.attachment_loss = AttachmentLoss()
        self.reset()

    def reset(self):
        self.n_words = 0
        self.n_err = 0

    def evaluate(self, example):
        self.n_err += self.attachment_loss.evaluate(example)
        self.n_words += example.N
        return self.n_err / max(1, self.n_words)
