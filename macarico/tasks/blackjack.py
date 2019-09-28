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
        macarico.Env.__init__(self, 2, 10)
        self.actions = [0, 1]
        self.payout_on_blackjack = payout_on_blackjack
        self._rewind()
        self.dealer = draw_hand()
        self.player = draw_hand()
        self.example.reward = 0
        
    def _rewind(self):
        self.dealer = draw_hand()
        self.player = draw_hand()
        self.example.reward = 0

    def _run_episode(self, policy):
        #print
        for self.t in range(self.horizon()):
            a = policy(self)
            #print self.dealer, self.player, a
            if a == 0: # hit
                self.player.append(draw_card())
                #print self.player
                if is_bust(self.player):
                    #print 'bust'
                    self.example.reward = 1
                    break
            else: # stay
                while sum_hand(self.dealer) < 17:
                    self.dealer.append(draw_card())
                #print self.dealer, score(self.player), score(self.dealer)
                if score(self.player) > score(self.dealer):
                    self.example.reward = -1
                    if self.payout_on_blackjack and is_blackjack(self.player):
                        self.example.reward = -1.5
                if score(self.player) < score(self.dealer): # technically this should be <= but using < for consistency
                    self.example.reward = 1
                #print reward
                break
        return self.output()

class BlackjackLoss(macarico.Loss):
    def __init__(self):
        super(BlackjackLoss, self).__init__('cost')
    def evaluate(self, example):
        return example.reward

class BlackjackFeatures(macarico.DynamicFeatures):
    def __init__(self):
        macarico.DynamicFeatures.__init__(self, 4)
        self._t = nn.Linear(1,1,bias=False)

    def _forward(self, state):
        view = util.zeros(self._t.weight, 1, 1, 4)
        view[0,0,0] = 1.
        view[0,0,1] = float(sum_hand(state.player))
        view[0,0,2] = float(state.dealer[0])
        view[0,0,3] = float(usable_ace(state.player))
        return Varng(view)
