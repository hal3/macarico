from __future__ import division
import random
import dynet as dy
import numpy as np

import macarico.util
macarico.util.reseed()

from macarico.lts.reinforce import Reinforce
from macarico.annealing import EWMA
from macarico.features.sequence import AttendAt
from macarico.features.actor import TransitionRNN, TransitionBOW
from macarico.policies.linear import LinearPolicy
from test_pocman import run_environment

from macarico.tasks.sysadmin import Network, SysAdmin, SysAdminLoss, SysAdminFeatures

net_size = 3

def run_sysadmin(net, actor):
    dy_model = dy.ParameterCollection()
    policy = LinearPolicy(dy_model, actor(dy_model), net_size+1)
    baseline = EWMA(0.8)
    optimizer = dy.AdamTrainer(dy_model, alpha=0.01)
    losses = []
    for epoch in xrange(3001):
        dy.renew_cg()
        learner = Reinforce(policy, baseline)
        env = net.mk_env()
        res,reward = env.run_episode(learner)
        loss = SysAdminLoss()(net, env)
        losses.append(np.sum(loss))
        if epoch % 10 == 0:
            print epoch, ' ', sum(losses[-500:]) / len(losses[-500:]), '\t', res, reward
        learner.update(loss)
        optimizer.update()
    

def test():
    print '\n===\n=== \n==='
    net = Network()
    run_sysadmin(
        net,
        lambda dy_model:
        TransitionBOW(dy_model,
                      [SysAdminFeatures()],
                      [AttendAt(lambda _: 0, 'computers')],
                      4)
    )

if __name__ == '__main__':
    test()
