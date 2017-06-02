from __future__ import division
import numpy as np
import random
import torch
import sys
import testutil
testutil.reseed()

from macarico.annealing import ExponentialAnnealing, stochastic
import macarico.lts.lols as LOLS
from macarico.lts.aggrevate import AggreVaTe
from macarico.tasks.sequence_labeler import Example, HammingLoss, HammingLossReference
from macarico.features.sequence import RNNFeatures, AttendAt
from macarico.features.actor import TransitionRNN
from macarico.policies.linear import LinearPolicy

def test1():
    n_types = 10
    n_labels = 4
    data = testutil.make_sequence_mod_data(20, 6, n_types, n_labels)
    data = [Example(x, y, n_labels) for x, y in data]

    tRNN = TransitionRNN([RNNFeatures(n_types)], [AttendAt()], n_labels)
    policy = LinearPolicy( tRNN, n_labels )
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    
    p_rollin_ref  = stochastic(ExponentialAnnealing(0.9))
    p_rollout_ref = stochastic(ExponentialAnnealing(0.9))

    testutil.trainloop(
        training_data   = data[:len(data)//2],
        dev_data        = data[len(data)//2:],
        policy          = policy,
        learning_alg    = lambda ex: LOLS.lols(ex, HammingLoss, HammingLossReference(), policy, p_rollin_ref, p_rollout_ref),
        losses          = HammingLoss(),
        optimizer       = optimizer,
        run_per_epoch   = [p_rollin_ref.step, p_rollout_ref.step],
        train_eval_skip = 1,
    )

def test2():
    # aggrevate
    print
    print '# test sequence labeler on mod data with DAgger'
    n_types = 10
    n_labels = 4

    data = testutil.make_sequence_mod_data(100, 5, n_types, n_labels)
    data = [Example(x, y, n_labels) for x, y in data]

    tRNN = TransitionRNN([RNNFeatures(n_types)],
                         [AttendAt()],
                         n_labels,
                        )
    policy = LinearPolicy(tRNN, n_labels)

    p_rollin_ref = stochastic(ExponentialAnnealing(0.99))
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)

    testutil.trainloop(
        training_data   = data[:len(data)//2],
        dev_data        = data[len(data)//2:],
        policy          = policy,
        Learner         = lambda: AggreVaTe(HammingLossReference(), policy, p_rollin_ref),
        losses          = HammingLoss(),
        optimizer       = optimizer,
        run_per_epoch   = [p_rollin_ref.step],
        n_epochs        = 4,
        train_eval_skip = 1,
    )
    
    
if __name__ == '__main__':
    test1()
    test2()
    
