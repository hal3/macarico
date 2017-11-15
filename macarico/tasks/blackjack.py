from __future__ import division, generators, print_function
import random
import macarico

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import macarico.util as util
from macarico.util import Var, Varng

deck = list(range(1,10)) + [10] * 4
draw_card = lambda: np.random.choice(deck)
draw_hand = lambda: [draw_card(), draw_card()]
def usable_ace(hand): return 1 in hand and sum(hand)+10 <= 21
def sum_hand(hand): return sum(hand) + (10 if usable_ace(hand) else 0)
def is_bust(hand): return sum_hand(hand) > 21
def score(hand): return 0 if is_bust(hand) else sum_hand(hand)
def is_blackjack(hand): return sorted(hand) == [1,10]

class Blackjack(macarico.Env):
    """
    largely based on the openai gym implementation
    """
    def __init__(self, payout_on_blackjack=False):
        self.actions = [0, 1]
        self.payout_on_blackjack = payout_on_blackjack
        self.dealer = []
        self.player = []
        self.T = 10
        self.n_actions = 2
        self.cost = 0
        
    def mk_env(self):
        self.dealer = draw_hand()
        self.player = draw_hand()
        self.cost = 0
        return self

    def _run_episode(self, policy):
        self._trajectory = []
        #print
        for self.t in range(self.T):
            a = policy(self)
            #print self.dealer, self.player, a
            self._trajectory.append(a)
            if a == 0: # hit
                self.player.append(draw_card())
                #print self.player
                if is_bust(self.player):
                    #print 'bust'
                    self.cost = 1
                    break
            else: # stay
                while sum_hand(self.dealer) < 17:
                    self.dealer.append(draw_card())
                #print self.dealer, score(self.player), score(self.dealer)
                if score(self.player) > score(self.dealer):
                    self.cost = -1
                    if self.payout_on_blackjack and is_blackjack(self.player):
                        self.cost = -1.5
                if score(self.player) < score(self.dealer): # technically this should be <= but using < for consistency
                    self.cost = 1
                #print reward
                break
        return self._trajectory

class BlackjackLoss(macarico.Loss):
    def __init__(self):
        super(BlackjackLoss, self).__init__('cost')
    def evaluate(self, ex, state):
        return state.cost

class BlackjackFeatures(macarico.StaticFeatures):
    def __init__(self):
        macarico.StaticFeatures.__init__(self, 4)
        self._t = nn.Linear(1,1,bias=False)

    def _forward(self, state):
        view = util.zeros(self._t.weight, 1, 1, 4)
        view[0,0,0] = 1.
        view[0,0,1] = float(sum_hand(state.player))
        view[0,0,2] = float(state.dealer[0])
        view[0,0,3] = float(usable_ace(state.player))
        return Varng(view)
