from __future__ import division
import numpy as np
import random
import torch
import sys

from macarico.annealing import ExponentialAnnealing, stochastic
import macarico.lts.lols as LOLS
from macarico.tasks.sequence_labeler import SequenceLabeling, BiLSTMFeatures, SeqFoci
from macarico import LinearPolicy

import testutil

def test1():
    n_types = 10
    n_labels = 4
    data = testutil.make_sequence_mod_data(20, 6, n_types, n_labels)
    # run
    policy = LinearPolicy(BiLSTMFeatures(SeqFoci(), n_types, n_labels,
                                         d_emb=10, d_actemb=4),
                          n_labels)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    
    p_rollin_ref  = stochastic(ExponentialAnnealing(0.9))
    p_rollout_ref = stochastic(ExponentialAnnealing(0.9))

    testutil.trainloop(
        Env             = lambda x: SequenceLabeling(x, n_labels),
        training_data   = data[:len(data)//2],
        dev_data        = data[len(data)//2:],
        policy          = policy,
#        Learner         = lambda ref: DAgger(ref, policy, p_rollin_ref),
        learning_alg    = lambda env, Y: LOLS.lols(env, Y, policy, p_rollin_ref, p_rollout_ref),
        optimizer       = optimizer,
        run_per_epoch   = [p_rollin_ref.step, p_rollout_ref.step],
        train_eval_skip = 1,
    )

    """
    train, dev = data[:len(data)//2], data[len(data)//2:]
    for epoch in xrange(100):
        for words, labels in train:
            env = SequenceLabeling(words, n_labels)
            optimizer.zero_grad()
            LOLS.lols(env, labels, policy, p_rollin_ref, p_rollout_ref)
            optimizer.step()

        print 'error rate: train %g dev %g' % \
            (evaluate(lambda w: SequenceLabeling(w, n_labels), train, policy),
             evaluate(lambda w: SequenceLabeling(w, n_labels), dev, policy))
    """

if __name__ == '__main__':
    test1()
    
