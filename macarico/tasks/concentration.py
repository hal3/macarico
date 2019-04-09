from __future__ import division, generators, print_function

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import macarico
import macarico.util as util
from macarico.util import Var, Varng
import numpy as np

class Concentration(macarico.Env):
    """
    similar, but simplified from, https://openreview.net/pdf?id=SJxE3jlA-
    """
    def __init__(self,
                 n_card_types=4,   # how many unique cards? (n_actions = 2*this)
                 random_deck_per_episode=True,  # draw new random "card images" for every episode?
                 max_horizon=None, # default to 5*n_card_types
                 ):
        # a reasonable policy would flip over every card (takes n_card_types steps)
        # and then would match them all (takes another n_card_types)
        self.n_card_types = n_card_types
        self.random_deck_per_episode = random_deck_per_episode
        self.faces = None
        self.similarity_threshold = 0.4
        macarico.Env.__init__(self,
                              2*n_card_types,
                              max_horizon or (5 * self.n_card_types))
        self.example.costs = []
        
    def _rewind(self):
        if self.faces is None or self.random_deck_per_episode:
            self.draw_new_faces()
        # card[i] is -1 if the card has already been removed; otherwise it's the face id
        self.n_remain = self.n_card_types
        self.card = np.random.permutation(list(range(self.n_card_types)) + list(range(self.n_card_types)) )

    def draw_new_faces(self):
        a = torch.rand((self.n_card_types, 6))
        for n in range(self.n_card_types):
            while torch.norm(a[n]) < self.similarity_threshold or \
                  (n > 0 and min([torch.norm(a[n] - a[i]) for i in range(n)]) < self.similarity_threshold):
                a[n] = torch.rand(6)
        self.faces = a
        
    def _run_episode(self, policy):
        self.flipped = None
        self.seen = set()
        self.card_seq = []
        self.example.costs = []
        
        for self.t in range(self.horizon()):
            # valid actions are un-flipped cards
            self.actions = set(np.where(self.card >= 0)[0])
            # if there's a flipped one, it's not a valid action
            if self.t % 2 == 1:
                self.actions.remove(self.flipped)
                
            a = policy(self)
            # make sure it's a valid card
            assert self.card[a] >= 0

            # update so we know we've seen this won
            self.seen.add(a)
            self.card_seq.append('%d:%d' % (a, self.card[a]))

            if self.t % 2 == 0:
                # this is the first card flipped, nothing happens
                self.example.costs.append(0.1)
            else:
                # this is the second card flipped
                if self.card[a] == self.card[self.flipped]: # match
                    self.card[a] = -1
                    self.card[self.flipped] = -1
                    self.n_remain -= 1
                    self.example.costs.append(-1 + 0.1)
                else:
                    self.example.costs.append(0.1)

            # update so we know which is the most recent card flipped
            self.flipped = a

            if self.n_remain == 0:
                break

        return self.output()

class ConcentrationLoss(macarico.Loss):
    def __init__(self): super(ConcentrationLoss, self).__init__('cost')
    def __call__(self, example): return sum(example.costs)
    
class ConcentrationPOFeatures(macarico.DynamicFeatures):
    def __init__(self):
        # features: the current face up card "image" (if flipped) otherwise 0s, and whether it's an odd or even turn
        super(ConcentrationPOFeatures, self).__init__(8)
        self._t = nn.Linear(1,1,bias=False)
        
    def _forward(self, state):
        c = util.zeros(self._t, 2)
        c[state.t % 2] = 1
        if state.flipped is None:
            return torch.cat([c, util.zeros(self._t, 6)]).view(1,1,-1)
        else:
            return torch.cat([c, state.faces[state.card[state.flipped]]]).view(1,1,-1)

    
class ConcentrationSmartFeatures(macarico.DynamicFeatures):
    def __init__(self, n_card_types, cheat=False):
        # features: 
        #   1. for each of the (2*n_card_types) positions:
        #        a one-hot of the card id that's there (if it's been seen yet) or 0 otherwise
        #   2. one-hot of the card id that's currently flipped
        #   3. one-hot for whether this is an even or odd flip
        #   4. CHEATING: one-hot for next card to flip per ConcentrationReference
        self.cheat = cheat
        self.dim = (2*n_card_types) * n_card_types + n_card_types + 2 + cheat * n_card_types * 2
        self.n_card_types = n_card_types
        super(ConcentrationSmartFeatures, self).__init__(self.dim)
        self._t = nn.Linear(1,1,bias=False)

    def _forward(self, state):
        f = []
        for n in range(2*self.n_card_types):
            c = util.zeros(self._t, self.n_card_types)
            if n in state.seen:
                c[state.card[n]] = 1
            f.append(c)
            
        c = util.zeros(self._t, self.n_card_types)
        if state.flipped is not None:
            c[state.card[state.flipped]] = 1
        f.append(c)
        
        c = util.zeros(self._t, 2)
        c[state.t % 2] = 1
        f.append(c)

        if self.cheat:
            c = util.zeros(self._t, 2*self.n_card_types)
            c[ConcentrationReference()(state)] = 1
            f.append(c)
        
        return torch.cat(f).view(1,1,-1)
    
class ConcentrationReference(macarico.Reference):
    def __call__(self, state):
        if state.t == 0:
            return 0 # first flip over the first card
        elif state.t % 2 == 1:
            # second flip; if we've seen a match to what's showing, flip it
            c = state.card[state.flipped]
            for i in state.seen:
                if i != state.flipped and c == state.card[i]:
                    return i
            # haven't seen this card before, which means there's at least one unseen, so flip it
            for i in range(len(state.card)):
                if i not in state.seen:
                    return i
            assert False
        else:
            # first flip; if we haven't flipped everything, flip the next one
            # otherwise flip the first remaining card
            first_remain = None
            for i in range(len(state.card)):
                if i not in state.seen:
                    return i
                elif state.card[i] >= 0 and first_remain is None:
                    first_remain = i
            assert first_remain is not None
            return first_remain
                
