from __future__ import division
import random
import torch
import numpy as np
import macarico.util
macarico.util.reseed()

from macarico.lts.dagger import DAgger
from macarico.annealing import ExponentialAnnealing
from macarico.tasks.sequence_labeler import Example, HammingLoss, HammingLossReference
from macarico.features.sequence import RNNFeatures, AttendAt
from macarico.features.actor import RNNActor
from macarico.policies.linear import LinearPolicy

def test0():
    print
    print '# test sequence labeler on mod data with DAgger'
    n_types = 10
    n_labels = 4

    data = [Example(x, y, n_labels) for x, y in macarico.util.make_sequence_mod_data(100, 5, n_types, n_labels)]

    features = RNNFeatures(n_types)
    attention = AttendAt(features, 'n')
    actor = RNNActor([attention], n_labels)
    policy = LinearPolicy(actor, n_labels)
    
    #tRNN = Actor(
    #             [RNNFeatures(
    #                          n_types,
    #                          output_field = 'mytok_rnn')],
    #             [AttendAt(field='mytok_rnn')],
    #             n_labels)
    #policy = LinearPolicy(tRNN, n_labels)

    learner = DAgger(HammingLossReference(), policy, ExponentialAnnealing(0.99))
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)

    macarico.util.trainloop(
        training_data   = data[:len(data)//2],
        dev_data        = data[len(data)//2:],
        policy          = policy,
        learner         = learner,
        losses          = HammingLoss(),
        optimizer       = optimizer,
        n_epochs        = 4,
        train_eval_skip = 1,
    )


if __name__ == '__main__':
    test0()
