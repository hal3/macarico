from __future__ import division
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var

import macarico.util
macarico.util.reseed()

from macarico.lts.reinforce import Reinforce
from macarico.annealing import EWMA
from macarico.tasks.gridworld import Example, GlobalGridFeatures, LocalGridFeatures, make_default_gridworld, make_big_gridworld, GridLoss
from macarico.features.sequence import AttendAt
from macarico.features.actor import TransitionRNN, TransitionBOW
from macarico.policies.linear import LinearPolicy

def run_gridworld(ex, actor):

    policy = LinearPolicy(actor(), 4)
    baseline = EWMA(0.8)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    losses = []
    for epoch in xrange(2001):
        dy.renew_cg()
        learner = Reinforce(policy, baseline)
        env = ex.mk_env()
        res = env.run_episode(learner)
        loss = GridLoss()(ex, env)
        losses.append(loss)
        if epoch % 100 == 0:
            print sum(losses[-10:]) / len(losses[-10:]), '\t', res
        learner.update(loss)
        optimizer.update()
    

def test0():
    print '\n===\n=== test0: p_step_success=1.0\n==='
    ex = make_default_gridworld(p_step_success=1.0)
    run_gridworld(
        ex,
        lambda dy_model:
        TransitionBOW(
                      [GlobalGridFeatures(ex.width, ex.height)],
                      [AttendAt(lambda _: 0, 'grid')],
                      4)
    )
    
def test1():
    print '\n===\n=== test1: p_step_success=0.8\n==='
    ex = make_default_gridworld(p_step_success=0.8)
    run_gridworld(
        ex,
        lambda dy_model:
        TransitionBOW(
                      [GlobalGridFeatures(ex.width, ex.height)],
                      [AttendAt(lambda _: 0, 'grid')],
                      4)
    )
    
def test2():
    print '\n===\n=== test2: p_step_success=0.8 and per_step_cost=0.1\n==='
    ex = make_default_gridworld(per_step_cost=0.1, p_step_success=0.8)
    run_gridworld(
        ex,
        lambda dy_model:
        TransitionBOW(
                      [GlobalGridFeatures(ex.width, ex.height)],
                      [AttendAt(lambda _: 0, 'grid')],
                      4)
    )

def test3():
    print '\n===\n=== test3: p_step_success=0.8, but local features only\n==='
    ex = make_default_gridworld(p_step_success=0.8, start_random=True)
    run_gridworld(
        ex,
        lambda dy_model:
        TransitionBOW(
                      [LocalGridFeatures(ex.width, ex.height)],
                      [AttendAt(lambda _: 0, 'grid')],
                      4)
    )

def test4():
    print '\n===\n=== test4: big grid world, global features\n==='
    ex = make_big_gridworld()
    run_gridworld(
        ex,
        lambda dy_model:
        TransitionBOW(
                      [GlobalGridFeatures(ex.width, ex.height)],
                      [AttendAt(lambda _: 0, 'grid')],
                      4)
    )
    

if __name__ == '__main__':
    #test0()
    #test1()
    #test2()
    test3()
    #test4()
