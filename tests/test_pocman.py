from __future__ import division
import random
import dynet as dy

import macarico.util
macarico.util.reseed()

from macarico.lts.reinforce import Reinforce
from macarico.annealing import EWMA
from macarico.features.sequence import AttendAt
from macarico.features.actor import TransitionRNN, TransitionBOW
from macarico.policies.linear import LinearPolicy
from macarico.tasks.pocman import MicroPOCMAN, MiniPOCMAN, FullPOCMAN, POCLoss, LocalPOCFeatures, GlobalPOCFeatures

def run_pocman(ex, actor):
    dy_model = dy.ParameterCollection()
    policy = LinearPolicy(dy_model, actor(dy_model), 4)
    baseline = EWMA(0.8)
    optimizer = dy.AdamTrainer(dy_model, alpha=0.01)
    losses = []
    for epoch in xrange(20001):
        dy.renew_cg()
        learner = Reinforce(policy, baseline)
        env = ex.mk_env()
        res = env.run_episode(learner, epoch % 500 == 0)
        loss = POCLoss()(ex, env)
        losses.append(loss)
        if epoch % 500 == 0:
            print sum(losses[-500:]) / len(losses[-500:]), '\t', res
        learner.update(loss)
        optimizer.update()
    
def test0():
    ex = FullPOCMAN()
    run_pocman(
        ex,
        lambda dy_model:
        TransitionBOW(dy_model,
                      [LocalPOCFeatures(history_length=4)], #ex.width, ex.height)],
                      [AttendAt(lambda _: 0, 'poc')],
                      4)
    )

test0()
