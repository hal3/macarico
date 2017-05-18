from __future__ import division

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
    This is an arc-hybrid dependency parser
    """

    SHIFT, RIGHT, LEFT, N_ACT = 0, 1, 2, 3

    def __init__(self, example, n_rels):
        self.example = example
        self.tokens = example.tokens
        self.pos = example.pos
        # TODO: add option for providing POS tags too
        # timv: will do this ^^ via Example class.
        self.N = len(self.tokens)
        self.i = 1
        self.a = None
        self.t = 0
        self.T = 2*self.N   # XXX: is this right???
        self.stack = [0]
        self.parse = ParseTree(self.N+1, n_rels>0)  # +1 for ROOT at end
        self.output = []
        self.actions = None
        self.n_rels = n_rels
        self.is_rel = None       # used to indicate whether the action type is a label action or not.
        self.n_actions = self.N_ACT + (self.n_rels or 0)
        if self.n_rels > 0:
            self.valid_rels = set(range(self.N_ACT, self.N_ACT+self.n_rels))

    def rewind(self):
        self.i = 1
        self.a = None
        self.t = 0
        self.stack = [0]
        self.parse = ParseTree(self.N+1)
        self.output = []
        self.actions = None

    def run_episode(self, policy):
        # run shift/reduce parser
        while self.stack or self.i+1 < self.N+1:  #n+1 for ROOT
            # get shift/reduce action
            self.is_rel = False
            self.actions = self.get_valid_transitions()
            #self.foci = [self.stack[-1], self.i]             # TODO: Create a DepFoci model.
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
                rel -= self.N_ACT

            self.transition(self.a, rel)

        return self.parse

    def get_valid_transitions(self):
        actions = set()
        if self.i+1 < self.N+1:  #n+1 for ROOT
            actions.add(self.SHIFT)
        stack_depth = len(self.stack)
        if stack_depth >= 2:
            actions.add(self.RIGHT)
        if stack_depth >= 1:
            actions.add(self.LEFT)
        return actions

    def transition(self, a, rel=None):
        if a == self.SHIFT:
            self.stack.append(self.i)
            self.i += 1
        elif a == self.RIGHT:
            self.parse.add(self.stack[-2], self.stack.pop(), rel)
        elif a == self.LEFT:
            self.parse.add(self.i, self.stack.pop(), rel)
        else:
            assert False, 'transition got invalid move %d' % a

    def loss(self):
        return AttachmentLoss(self)()

    def reference(self):
        return AttachmentLoss(self).reference()


class AttachmentLossReference(macarico.Reference):
    def __init__(self, env):
        self.true_heads = env.example.heads
        self.true_rels  = env.example.rels

    def __call__(self, state):
        if state.is_rel:
            return random.choice(self.relation_reference(state))
        else:
            ref = self.transition_reference(state)
            ## debug set_min_costs_to_go
            if False:
                costs = np.zeros(3)
                self.transition_costs(state, costs)
                assert all((costs[r] == costs[ref[0]] for r in ref)) and \
                       all((costs[ref[0]] <= c for a, c in enumerate(costs) if a not in ref)), \
                    'reference failed, ref=%s costs=%s\nstack = %s\ntrue_heads = %s\nidx = %s\nN = %s' % (ref, costs, state.stack, self.true_heads, state.i, state.N)
            return random.choice(ref)

    def set_min_costs_to_go(self, state, cost_vector):
        if state.is_rel:
            ref = self.relation_reference(state)
            cost_vector *= 0
            cost_vector += 1
            if len(ref) == 1:  # this is the correct relation
                cost_vector[ref[0]] = 0.
            # otherwise anything goes and all costs are 1
        else:  # predicting action
            self.transition_costs(state, cost_vector)

    def relation_reference(self, state):
        a = state.a
        if a == state.RIGHT:
            # new edge is parse.add(state.stack[-2], state.stack.pop(), rel)
            head  = state.stack[-2]
            child = state.stack[-1]
        elif a == state.LEFT:
            # new edge is parse.add(state.i, state.stack.pop(), rel)
            head  = state.i
            child = state.stack[-1]
        else:
            assert False, 'relation_reference called with a=%s was neither LEFT nor RIGHT' % a

        if self.true_heads[child] == head:
            return [self.true_rels[child] + state.N_ACT]
        else:
            return list(state.actions)

    def transition_costs(self, state, costs):
        costs[state.SHIFT] = 0
        costs[state.RIGHT] = 0
        costs[state.LEFT]  = 0

        stack = state.stack
        size = len(stack)
        last = stack[-1] if size>0 else 0
        true_heads = self.true_heads
        idx = state.i
        N = state.N

        if idx < N:
            for i in stack:
                if true_heads[i] == idx or true_heads[idx] == i:
                    costs[state.SHIFT] += 1

        if size > 0 and true_heads[last] == idx:
            costs[state.SHIFT] += 1

        for i in xrange(idx+1, N):
            if true_heads[i] == last or true_heads[last] == i:
                costs[state.LEFT] += 1

        if size > 0 and idx < N and true_heads[idx] == last:
            costs[state.LEFT] += 1

        if size > 1 and true_heads[last] == stack[-2]:
            costs[state.LEFT] += 1

        if true_heads[last] >= idx and true_heads[last] < N:
            costs[state.RIGHT] += 1

        for i in xrange(idx, N):
            if true_heads[i] == last:
                costs[state.RIGHT] += 1
        
    def transition_reference(self, state):
        stack = state.stack
        true_heads = self.true_heads
        i = state.i
        N = state.N

        def deps_between(target, others):
            return any((true_heads[j] == target or true_heads[target] == j for j in others))

        if (not stack
            or (state.SHIFT in state.actions
                and true_heads[i] == stack[-1])):
            return [state.SHIFT]

        if true_heads[stack[-1]] == i:
            return [state.LEFT]

        costly = set()
        if len(stack) >= 2 and true_heads[stack[-1]] == stack[-2]:
            costly.add(state.LEFT)

        if state.SHIFT in state.actions and deps_between(i, stack):
            costly.add(state.SHIFT)

        if deps_between(stack[-1], xrange(i+1, N)):
            costly.add(state.LEFT)
            costly.add(state.RIGHT)

        return [m for m in state.actions if m not in costly]
    
class AttachmentLoss(object):
    def __init__(self, env): #true_heads, true_rels=None):
        self.env = env
        self.true_heads = env.example.heads
        self.true_rels = env.example.rels

    def __call__(self):
        loss = 0
        for n,head in enumerate(self.true_heads):
            if self.env.parse.heads[n] != head:
                loss += 1
            elif self.true_rels is not None and \
                 self.env.parse.rels[n] != self.true_rels[n]:
                loss += 1
        return loss

    def reference(self):
        return AttachmentLossReference(self.env)


class DepParFoci:
    arity = 2
    def __init__(self, field='tokens_rnn'):
        self.field = field
    def __call__(self, state):
        buffer_pos = state.i if state.i < state.N else None
        stack_pos  = state.stack[-1] if state.stack else None
        #print '[foci=%s]' % [buffer_pos, stack_pos],
        return [buffer_pos, stack_pos]
