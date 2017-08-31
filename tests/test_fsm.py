from __future__ import division
import random
import dynet as dy
import numpy as np

import macarico.util
macarico.util.reseed()

from macarico.annealing import ExponentialAnnealing, stochastic
from macarico.lts.maximum_likelihood import MaximumLikelihood
from macarico.lts.dagger import DAgger, TwistedDAgger
from macarico.lts.aggrevate import AggreVaTe
from macarico.lts.lols import lols
from macarico.annealing import EWMA
from macarico.features.actor import TransitionRNN, TransitionBOW
from macarico.features.sequence import AttendAt
from macarico.policies.linear import LinearPolicy
from macarico.tasks import fsm

class LearnerOpts:
    MAXLIK = 'MaximumLikelihood'
    DAGGER = 'DAgger'
    TWISTED = 'TwistedDAgger'
    AGGREVATE = 'AggreVaTe'
    LOLS = 'LOLS'

def make_ross_fsm(T=100, reset_prob=0):
    initial = [(0, 1/3), (1, 1/3)]
    #               s    a    s' p()
    transitions = { 0: { 0: [(1, 1.0-reset_prob), (0, reset_prob/2), (2, reset_prob/2)],
                         1: [(2, 1.0-reset_prob), (0, reset_prob/2), (1, reset_prob/2)] },
                    1: { 0: [(2, 1.0-reset_prob), (0, reset_prob/2), (1, reset_prob/2)],
                         1: [(1, 1.0-reset_prob), (0, reset_prob/2), (2, reset_prob/2)] },
                    2: { 0: [(1, 1.0-reset_prob), (1, reset_prob/2), (2, reset_prob/2)],
                         1: [(2, 1.0-reset_prob), (0, reset_prob/2), (2, reset_prob/2)] } }

    def pi_ref(s):
        if isinstance(s, fsm.FSM):
            s = s.s
        # expert: s0->a0 s1->a1 s2->a0
        if s == 0: return 0
        if s == 1: return 1
        if s == 2: return 0
        assert False
        
    def costs(s, a, s1):
        # this is just Cmax=1 whenever we disagree with expert, and c=0 otherwise
        return 0 if a == pi_ref(s) else 1
    
    return fsm.FSMExample(initial, transitions, costs, T), \
           fsm.DeterministicReference(pi_ref)
    
def test1(LEARNER=LearnerOpts.DAGGER):
    print
    print 'Running test 1 with learner=%s' % LEARNER
    print '======================================================='

    dy_model = dy.ParameterCollection()

    n_states = 3
    n_actions = 2
    
    tRNN = TransitionRNN(dy_model,
                         [fsm.FSMFeatures(n_states, noise_rate=0.5)],
                         [AttendAt(lambda _: 0, 's')],
                         n_actions)
    policy = LinearPolicy(dy_model, tRNN, n_actions)

    p_rollin_ref  = stochastic(ExponentialAnnealing(0.99))
    p_rollout_ref  = stochastic(ExponentialAnnealing(1))

    optimizer = dy.AdamTrainer(dy_model, alpha=0.001)

    test_fsm, pi_ref = make_ross_fsm()

    if LEARNER == LearnerOpts.DAGGER:
        learner = lambda: DAgger(pi_ref, policy, p_rollin_ref)
    elif LEARNER == LearnerOpts.TWISTED:
        learner = lambda: TwistedDAgger(pi_ref, policy, p_rollin_ref)
    elif LEARNER == LearnerOpts.MAXLIK:
        learner = lambda: MaximumLikelihood(pi_ref, policy)
    elif LEARNER == LearnerOpts.AGGREVATE:
        learner = lambda: AggreVaTe(pi_ref, policy, p_rollin_ref)
    elif LEARNER == LearnerOpts.LOLS:
        learner = None

    losses = []
    for epoch in xrange(2001):
        dy.renew_cg()
        if learner is not None:
            l = learner()
            env = test_fsm.mk_env()
            res = env.run_episode(l)
            loss = fsm.FSMLoss()(test_fsm, env)
            l.update(loss)
        elif LEARNER == LearnerOpts.LOLS:
            lols(test_fsm, fsm.FSMLoss, pi_ref, policy, p_rollin_ref, p_rollout_ref)
        
        optimizer.update()
        p_rollin_ref.step()
        p_rollout_ref.step()

        env = test_fsm.mk_env()
        res = env.run_episode(policy)
        loss = fsm.FSMLoss()(test_fsm, env)
        losses.append(loss)
        if epoch % 200 == 0:
            print epoch, sum(losses[-100:]) / len(losses[-100:]), '\t', res

if __name__ == '__main__':
    #test1(LearnerOpts.MAXLIK)
    #test1(LearnerOpts.DAGGER)
    #test1(LearnerOpts.TWISTED)
    #test1(LearnerOpts.AGGREVATE)
    test1(LearnerOpts.LOLS)
