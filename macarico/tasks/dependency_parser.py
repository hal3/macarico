from __future__ import division
from collections import defaultdict
import random
import numpy as np
import macarico

class Example(object):

    def __init__(self, tokens, heads, rels, n_rels, pos=None):
        self.tokens = tokens
        self.pos = pos
        self.heads = heads
        self.rels = rels
        self.n_rels = n_rels

    def mk_env(self):
        return DependencyParser(self, self.n_rels)

    def __str__(self):
        return str(self.heads)

class ParseTree(object):

    def __init__(self, n, labeled=False):
        self.n = n
        self.labeled = labeled
        self.heads = [None] * (n-1)   # TODO: we should probably just hard code a root token, no?
        self.rels = ([None] * (n-1)) if labeled else None

    def add(self, head, child, rel=None):
        #print 'head[%d] <- %d' % (child, head)
        self.heads[child] = head
        if self.labeled:
            self.rels[child] = rel

    def __repr__(self):
        s = 'heads = %s' % str(self.heads)
        if self.labeled:
            s += '\nrels  = %s' % str(self.rels)
        return s

    def __str__(self):
        S = []
        for i in xrange(self.n-1):
            x = '%d->%s' % (i, self.heads[i])
            if self.labeled:
                x = '%s[%s]' % (x, self.rels[i])
            S.append(x)
        return ' '.join(S)
#        return str(self.heads)


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

    def __init__(self, example, n_rels):
        self.example = example
        self.tokens = example.tokens
        self.pos = example.pos
        self.N = len(self.tokens)

        self.gold_heads = {}
        self.gold_deps = defaultdict(lambda: [])
        if self.example.heads is not None:
            for dep in range(self.N):
                head = self.example.heads[dep]
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
        self.t = 0
        self.T = 2*self.N   # XXX: is this right???
        self.parse = ParseTree(self.N+1, n_rels>0)  # +1 for ROOT at end
        self.output = []
        self.actions = None
        self.n_rels = n_rels
        self.is_rel = None       # used to indicate whether the action type is a label action or not.
        if self.n_rels > 0:
            self.valid_rels = set(range(self.N_ACT, self.N_ACT+self.n_rels))
            self.T *= 2
        super(DependencyParser, self).__init__(self.N_ACT + (self.n_rels or 0))

    def rewind(self):
        #print '\n-------------------'
        self.a = None
        self.t = 0
        self.stack = []
        self.b = 0
        self.parse = ParseTree(self.N+1, self.n_rels > 0)
        self.output = []
        self.actions = None

    def run_episode(self, policy):
        # run shift/reduce parser
        while True: #not (len(self.stack) == 0 and len(self.buf) == 1):
#            assert self.buf == range(self.b, self.N+1), \
#                'b=%d buf=%s' % (self.b, self.buf)
            # len(self.buf) == self.N - self.b+1
            # so len(buf) > 0 <==> self.N+1 - self.b > 0 <==> self.b < self.N+1 <==> self.b <= self.N
            if len(self.stack) == 0 and self.b == self.N:
                break
            # get shift/reduce action
            #print 'stack = %s\tbuf = %s' % (self.stack, self.b)
            self.is_rel = False
            self.actions = self.get_valid_transitions()
            #print 'actions = %s' % self.actions
            self.a = policy(self)
            #print 'i=%d\tstack=%s\tparse=%s\ta=%s' % (self.i, self.stack, self.parse, self.a),
            assert self.a in self.actions, 'policy %s returned an invalid transition "%s"!' % (type(policy), self.a)
            self.output.append(self.a)
            self.t += 1

            # if we're doing labeled parsing, get relation
            rel = None
            if self.n_rels > 0 and self.a != self.SHIFT:
                self.is_rel = True
                self.actions = self.valid_rels
                rel = policy(self)
                assert rel is not None
                #if rel is None:   # timv: @hal3 why will this ever be None?
                #    rel = random.choice(self.valid_rels)
                self.output.append(rel)
                rel -= self.N_ACT
                self.t += 1

            self.transition(self.a, rel)

        return self.parse

    def get_valid_transitions(self):
        actions = set()
        
        if self.b < self.N: # len(self.buf) > 1:
            actions.add(self.SHIFT)
            
        if len(self.stack) >= 2:
            actions.add(self.RIGHT)
            
        if self.b <= self.N and len(self.stack) > 0 and self.stack[-1] != self.root:
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
            self.parse.add(s1, s0, rel)
        elif a == self.LEFT: # == 2
            # in KW's code, heads[stack[-1]] = idx
            # so stack[-1] = child, idx = head
            s0 = self.stack.pop()
            #b = self.buf[0]
            self.parse.add(self.b, s0, rel)
        else:
            assert False, 'transition got invalid move %d' % a


class AttachmentLossReference(macarico.Reference):
    def __init__(self):
        pass

    def __call__(self, state):
        if state.is_rel:
            return random.choice(self.relation_reference(state))
        else:
            costs = np.zeros(3) + 999
            for a in state.actions: costs[a] = 0
            self.transition_costs(state, costs)
            #print costs
            ref = None
            for a in state.actions:
                if ref is None or costs[a] < costs[ref]:
                    ref = a
            return ref
            
    def transition_costs(self, state, costs):
        # SHIFT: then b=buf[0] will be put onto the stack, and won't
        # be able to get heads from {s1}+S and will not be able to get
        # deps from {s0,s1}+S
        if state.b <= state.N and len(state.stack) > 0:
            if state.b in state.gold_heads:  # no
                for j in state.stack[0:-1]:
                    if j == state.gold_heads[state.b]:
                        costs[state.SHIFT] += 1
            for dep in state.gold_deps[state.b]: # deps[2] = [0, 1]
                if dep in state.stack: # stack = [0, 1], so YES
                    costs[state.SHIFT] += 1

        # RIGHT: adding arc (s1,s0) and popping s0 means s0 won't be
        # able to acquire heads or deps from B
        if len(state.stack) > 0:
            s0 = state.stack[-1]
            for b in range(state.b, state.N+1):
                if (b in state.gold_heads and state.gold_heads[b] == s0) or \
                   state.gold_heads[s0] == b:
                    costs[state.RIGHT] += 1
                    
        # LEFT: adding arc (b,s0) and popping s0 from stack means s0
        # won't be able to acquire heads from {s1}+B nor dependents
        # from B+b.
        if len(state.stack) > 1:
            s0 = state.stack[-1]
            s1 = state.stack[-2]
            if state.gold_deps[s0] in range(state.b, state.N+1):
                costs[state.LEFT] += 1

            #H = state.buf[1:] + [s1]
            if s0 in state.gold_heads:
                #if state.gold_heads[s0] in H:
                if state.gold_heads[s0] > state.b:
                    costs[state.LEFT] += 1
                if state.gold_heads[s0] == s1:
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
            return [state.example.rels[child] + state.N_ACT]
        else:
            return list(state.actions)

class AttachmentLoss(macarico.Loss):
    def __init__(self):
        super(AttachmentLoss, self).__init__('lal')

    def evaluate(self, ex, state):
        loss = 0
        for n,head in enumerate(ex.heads):
            if state.parse.heads[n] != head:
                #print 'err on n=%d, head=%d, parse.head=%d' % (n, head, state.parse.heads[n])
                loss += 1
            elif ex.rels is not None and state.parse.rels[n] != ex.rels[n]:
                loss += 1
        return loss


class DependencyAttention(macarico.Attention):
    arity = 3
    def __init__(self, field='tokens_feats'):
        super(DependencyAttention, self).__init__(field)

    def __call__(self, state):
        buffer_pos = state.b if state.b < state.N else None   # for shift
        stack_pos  = state.stack[-1] if len(state.stack) > 0 else None # for left & right
        stack_pos2 = state.stack[-2] if len(state.stack) > 1 else None # for right
        return [buffer_pos, stack_pos, stack_pos2]
