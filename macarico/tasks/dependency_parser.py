from __future__ import division

import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import macarico

zeros  = lambda d: Variable(torch.zeros(1,d))
onehot = lambda i: Variable(torch.LongTensor([i]))

class ParseTree(object):
    def __init__(self, n):
        self.n = n
        self.heads = [None] * (n-1)
        self.rels = [None] * (n-1)

    def add(self, head, child, rel=None):
        if head == self.n-1: head = None
        self.heads[child] = head
        self.rels[child] = rel

    def __repr__(self):
        s = 'heads = %s' % str(self.heads)
        if any((l is not None for l in self.rels)):
            s += '\nrels  = %s' % str(self.rels)
        return s

    def __str__(self):
        """
        s = ''
        for i in range(self.n-1):
            s += '%d->%s' % (i, self.heads[i])
            if self.rels[i] is not None:
                s += '[%d]' % self.rels[i]
            s += ' '
        return s[:-1]
        """
        return str(self.heads)

class DependencyParser(macarico.Env):
    """
    A greedy transition-based parser, based heavily on
    Matthew Honnibal's "500 lines" implementation:
      https://gist.github.com/syllog1sm/10343947
    """

    SHIFT, RIGHT, LEFT, N_ACT = 0, 1, 2, 3

    def __init__(self, tokens, n_rels=0):
        # TODO: add option for providing POS tags too
        self.tokens = tokens
        self.N = len(self.tokens)
        self.i = 1
        self.a = None
        self.t = 0
        self.stack = [0]
        self.parse = ParseTree(self.N+1)  # +1 for ROOT at end
        self.prev_action = None
        self.n_rels = n_rels
        if self.n_rels > 0:
            self.valid_rels = range(DependencyParser.N_ACT, DependencyParser.N_ACT+self.n_rels)

    def run_episode(self, policy):
        # run shift/reduce parser
        while self.stack or self.i+1 < self.N+1:  #n+1 for ROOT
            # get shift/reduce action
            valid_transitions = self.get_valid_transitions()
            #self.foci = [self.stack[-1], self.i]             # TODO: Create a DepFoci model.
            self.t += 1
            self.a = policy(self, limit_actions=valid_transitions)
            if isinstance(self.a, list):
                self.a = random.choice(self.a)
            assert self.a in valid_transitions, 'policy %s returned an invalid transition "%s"!' % (type(policy), self.a)
            self.prev_action = self.a

            # if we're doing labeled parsing, get relation
            rel = None
            if self.n_rels > 0 and self.a != DependencyParser.SHIFT:
                rel = policy(self, limit_actions=self.valid_rels)
                if rel is None:
                    rel = random.choice(self.valid_rels)
                rel -= DependencyParser.N_ACT

            self.transition(self.a, rel)

        return self.parse

    def get_valid_transitions(self):
        actions = set()
        if self.i+1 < self.N+1:  #n+1 for ROOT
            actions.add(DependencyParser.SHIFT)
        stack_depth = len(self.stack)
        if stack_depth >= 2:
            actions.add(DependencyParser.RIGHT)
        if stack_depth >= 1:
            actions.add(DependencyParser.LEFT)

        return actions

    def transition(self, a, rel=None):
        if a == DependencyParser.SHIFT:
            self.stack.append(self.i)
            self.i += 1
        elif a == DependencyParser.RIGHT:
            self.parse.add(self.stack[-2], self.stack.pop(), rel)
        elif a == DependencyParser.LEFT:
            self.parse.add(self.i, self.stack.pop(), rel)
        else:
            assert False, 'transition got invalid move %d' % a

    def loss_function(self, true_heads, true_rels=None):
        return AttachmentLoss(self, true_heads, true_rels)

class AttachmentLoss(object):
    def __init__(self, env, true_heads, true_rels=None):
        self.env = env
        self.true_heads = true_heads
        self.true_rels = true_rels

    def __call__(self):
        loss = 0
        for n,head in enumerate(self.true_heads):
            if self.env.parse.heads[n] != head:
                loss += 1
            elif self.true_rels is not None and \
                 self.env.parse.rels[n] != self.true_rels[n]:
                loss += 1
        return loss

    def reference(self, state, limit_actions=None):
        if limit_actions is None:
            limit_actions = state.get_valid_transitions()
        is_trans = 0 in limit_actions or 1 in limit_actions or 2 in limit_actions
        is_rel   = DependencyParser.N_ACT in limit_actions
        assert is_trans != is_rel, 'reference limite_actions contains both transition and relation actions'
        if is_trans:
            return self.transition_reference(state, limit_actions)
        if is_rel:
            return self.relation_reference(state, limit_actions)
        assert False, 'should be impossible to get here!'

    def relation_reference(self, state, limit_actions):
        a = state.a
        if a == DependencyParser.RIGHT:
            # new edge is parse.add(state.stack[-2], state.stack.pop(), rel)
            head  = state.stack[-2]
            child = state.stack[-1]
        elif a == DependencyParser.LEFT:
            # new edge is parse.add(state.i, state.stack.pop(), rel)
            head  = state.i
            child = state.stack[-1]
        else:
            assert False, 'relation_reference called with a=%s was neither LEFT nor RIGHT' % a

        if self.true_heads[child] == head:
            return self.true_rels[child] + DependencyParser.N_ACT
        else:
            return None

    def transition_reference(self, state, limit_actions):
        stack = state.stack
        true_heads = self.true_heads
        i = state.i
        N = state.N

        def deps_between(target, others):
            return any((true_heads[j] == target or true_heads[target] == j for j in others))

        if not stack or \
           (DependencyParser.SHIFT in limit_actions and \
            true_heads[i] == stack[-1]):
            return [DependencyParser.SHIFT]

        if true_heads[stack[-1]] == i:
            return [DependencyParser.LEFT]

        costly = set()
        if len(stack) >= 2 and true_heads[stack[-1]] == stack[-2]:
            costly.add(DependencyParser.LEFT)

        if DependencyParser.SHIFT in limit_actions and deps_between(i, stack):
            costly.add(DependencyParser.SHIFT)

        if deps_between(stack[-1], range(i+1, N-1)):
            costly.add(DependencyParser.LEFT)
            costly.add(DependencyParser.RIGHT)

        return [m for m in limit_actions if m not in costly]

