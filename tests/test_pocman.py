from __future__ import division
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
from macarico.tasks.hex import Hex, HexLoss, HexFeatures
from macarico.lts.reinforce import AdvantageActorCritic, LinearValueFn

def run_environment(ex, actor, lossfn, rl_alg=lambda x: Reinforce(x[1]), n_epochs=2000, lr=0.01):

    policy = LinearPolicy(actor(), ex.n_actions, n_layers=1)
    #baseline = LinearValueFn(policy.features.dim)
    #baseline = EWMA(0.8)
    #optimizer = dy.SimpleSGDTrainer(learning_rate=lr)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    #optimizer = dy.AdagradTrainer(learning_rate=lr)
    #optimizer = dy.AdadeltaTrainer()
    #optimizer = dy.RMSPropTrainer(learning_rate=lr)
    #optimizer = dy.MomentumSGDTrainer(learning_rate=lr)
    #optimizer.set_clip_threshold(0.)
    losses = []
    for epoch in xrange(n_epochs):
        dy.renew_cg()
        learner = rl_alg(policy)
        #learner = AdvantageActorCritic(policy, baseline)
        env = ex.mk_env()
        res = env.run_episode(learner) # , epoch % 5000 == 0)
        loss = lossfn(ex, env)
        losses.append(loss)
        if epoch % 1000 == 0:
            print epoch, '\t', sum(losses[-500:]) / len(losses[-500:]), '\t', res
        learner.update(loss)
        optimizer.update()

def test0():
    print
    print 'micro pocman'
    print
    ex = MicroPOCMAN()
    run_environment(
        ex,
        lambda dy_model:
        TransitionBOW(
                      [LocalPOCFeatures(history_length=4)], #ex.width, ex.height)],
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
        lambda dy_model:
        TransitionRNN(
                      [PendulumFeatures()], #ex.width, ex.height)],
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
        lambda dy_model:
        TransitionBOW(
                      [BlackjackFeatures()], #ex.width, ex.height)],
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
        lambda dy_model:
        TransitionBOW(
                      [HexFeatures(board_size)], #ex.width, ex.height)],
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
