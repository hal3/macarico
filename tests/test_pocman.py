from __future__ import division, generators, print_function
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var

import macarico.util
macarico.util.reseed()

from macarico.lts.reinforce import Reinforce
from macarico.annealing import EWMA
from macarico.features.sequence import AttendAt
from macarico.features.actor import TransitionRNN, TransitionBOW
from macarico.policies.linear import LinearPolicy
from macarico.tasks.pocman import MicroPOCMAN, MiniPOCMAN, FullPOCMAN, POCLoss, LocalPOCFeatures, GlobalPOCFeatures, POCReference
from macarico.tasks.pendulum import Pendulum, PendulumLoss, PendulumFeatures
from macarico.tasks.blackjack import Blackjack, BlackjackLoss, BlackjackFeatures
from macarico.tasks.hexgame import Hex, HexLoss, HexFeatures
from macarico.lts.reinforce import AdvantageActorCritic, LinearValueFn

def run_environment(ex, actor, lossfn, rl_alg=None, n_epochs=201, lr=0.01):
    if rl_alg is None:
        baseline = EWMA(0.8)
        rl_alg = lambda policy: Reinforce(policy, baseline)
    policy = LinearPolicy(actor(), ex.n_actions, n_layers=1)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    losses = []
    for epoch in xrange(n_epochs):
        optimizer.zero_grad()
        learner = rl_alg(policy)
        #learner = AdvantageActorCritic(policy, baseline)
        env = ex.mk_env()
        res = env.run_episode(learner) # , epoch % 5000 == 0)
        loss = lossfn(ex, env)
        losses.append(loss)
        if epoch % 20 == 0:
            print epoch, '\t', sum(losses[-500:]) / len(losses[-500:]), '\t', res
        learner.update(loss)
        optimizer.step()

def test0():
    print
    print 'micro pocman'
    print
    ex = MicroPOCMAN()
    run_environment(
        ex,
        lambda:
        TransitionBOW([LocalPOCFeatures(history_length=4)], #ex.width, ex.height)],
                      [AttendAt(lambda _: 0, 'poc')],
                      4),
        POCLoss(),
    )

def test1():
    print
    print 'pendulum'
    print
    ex = Pendulum()
    run_environment(
        ex,
        lambda:
        TransitionRNN([PendulumFeatures()], #ex.width, ex.height)],
                      [AttendAt(lambda _: 0, 'pendulum')],
                      ex.n_actions),
        PendulumLoss(),
    )

def test2():
    print
    print 'blackjack'
    print
    ex = Blackjack()
    run_environment(
        ex,
        lambda:
        TransitionBOW([BlackjackFeatures()], #ex.width, ex.height)],
                      [AttendAt(lambda _: 0, 'blackjack')],
                      ex.n_actions),
        BlackjackLoss(),
    )

def test3():
    print
    print 'hex'
    print
    board_size = 3
    ex = Hex(Hex.BLACK, board_size)
    run_environment(
        ex,
        lambda:
        TransitionBOW([HexFeatures(board_size)], #ex.width, ex.height)],
                      [AttendAt(lambda _: 0, 'hex')],
                      ex.n_actions),
        HexLoss(),
    )

def test_ref():
    env = MicroPOCMAN().mk_env()
    env.run_episode(POCReference(), True)
    loss = POCLoss()(env, env)
    print 'loss =', loss

#test_ref()
#time.sleep(5)
if __name__ == '__main__':
    test0()
    test1()
    test2()
    test3()
