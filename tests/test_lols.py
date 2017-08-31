from __future__ import division
import numpy as np
import random
import dynet as dy
import sys
import macarico.util
macarico.util.reseed()

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
    print
    print '# test sequence labeler on mod data with LOLS'
    data = macarico.util.make_sequence_mod_data(20, 6, n_types, n_labels)
    data = [Example(x, y, n_labels) for x, y in data]

    dy_model = dy.ParameterCollection()
    tRNN = TransitionRNN(dy_model, [RNNFeatures(dy_model, n_types)], [AttendAt()], n_labels)
    policy = LinearPolicy(dy_model, tRNN, n_labels)
    optimizer = dy.AdamTrainer(dy_model, alpha=0.01)
    
    p_rollin_ref  = stochastic(ExponentialAnnealing(0.9))
    p_rollout_ref = stochastic(ExponentialAnnealing(0.9))

    macarico.util.trainloop(
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
    print '# test sequence labeler on mod data with AggreVaTe'
    n_types = 10
    n_labels = 4

    data = macarico.util.make_sequence_mod_data(100, 5, n_types, n_labels)
    data = [Example(x, y, n_labels) for x, y in data]

    dy_model = dy.ParameterCollection()
    tRNN = TransitionRNN(dy_model, [RNNFeatures(dy_model, n_types)],
                         [AttendAt()],
                         n_labels,
                        )
    policy = LinearPolicy(dy_model, tRNN, n_labels)

    p_rollin_ref = stochastic(ExponentialAnnealing(0.99))
    optimizer = dy.AdamTrainer(dy_model, alpha=0.01)

    macarico.util.trainloop(
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
    
