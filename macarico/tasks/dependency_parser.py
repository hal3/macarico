from __future__ import division

import random
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
        if self.n_rels > 0:
            self.valid_rels = range(self.N_ACT, self.N_ACT+self.n_rels)

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
            if isinstance(self.a, list):    # TODO: timv: I don't think we should let policies return lists. For non det oracles, we should just have them break ties (e.g., by randomness or with the learned policy)
                self.a = random.choice(self.a)
#            assert self.a in valid_transitions, 'policy %s returned an invalid transition "%s"!' % (type(policy), self.a)
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
        return AttachmentLoss(self).reference


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

    def reference(self, state):
        if state.is_rel:
            return self.relation_reference(state)
        else:
            return self.transition_reference(state)

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
            return self.true_rels[child] + state.N_ACT
        else:
            return random.choice(state.actions)

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

        if deps_between(stack[-1], range(i+1, N)):
            costly.add(state.LEFT)
            costly.add(state.RIGHT)

        return [m for m in state.actions if m not in costly]


class DepParFoci:
    arity = 2
    def __init__(self, field='tokens_rnn'):
        self.field = field
    def __call__(self, state):
        buffer_pos = state.i if state.i < state.N else None
        stack_pos  = state.stack[-1] if state.stack else None
        #print '[foci=%s]' % [buffer_pos, stack_pos],
        return [buffer_pos, stack_pos]
