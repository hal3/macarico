from __future__ import division
import random
import torch
import testutil
testutil.reseed()

from macarico.annealing import ExponentialAnnealing, stochastic
from macarico.lts.maximum_likelihood import MaximumLikelihood
from macarico.lts.dagger import DAgger
from macarico.tasks.seq2seq import Seq2Seq, FrontBackAttention, Example
from macarico.features.sequence import RNNFeatures
from macarico.features.actor import TransitionRNN
from macarico.policies.linear import LinearPolicy

def test1():
    n_types = 8
    data = testutil.make_sequence_reversal_data(100, 3, n_types)
    # make EOS=0 and add one to all outputs
    n_labels = 1 + n_types
    data = [Example(X, [y+1 for y in Y] + [0], n_labels) \
            for X,Y in data]

    tRNN = TransitionRNN([RNNFeatures(n_types)], [FrontBackAttention()], n_labels)
    policy = LinearPolicy( tRNN, n_labels )
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    p_rollin_ref = stochastic(ExponentialAnnealing(0.99))
    
    print 'eval ref: %s' % testutil.evaluate(data, lambda s: s.reference()(s))
    
    testutil.trainloop(
        training_data   = data[:len(data)//2],
        dev_data        = data[len(data)//2:],
        policy          = policy,
        Learner         = lambda ref: DAgger(ref, policy, p_rollin_ref),
        optimizer       = optimizer,
        train_eval_skip = 1,
        n_epochs        = 40,
    )


if __name__ == '__main__':
    test1()
