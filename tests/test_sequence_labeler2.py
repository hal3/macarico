from __future__ import division, generators, print_function
import random
import torch
import numpy as np
import macarico.util
macarico.util.reseed()

from macarico.lts.dagger import DAgger
from macarico.annealing import ExponentialAnnealing
from macarico.tasks.sequence_labeler import Example, HammingLoss, HammingLossReference
from macarico.features.sequence import EmbeddingFeatures, BOWFeatures, RNN, DilatedCNN, AttendAt, FrontBackAttention, SoftmaxAttention
from macarico.actors.rnn import RNNActor
from macarico.actors.bow import BOWActor
from macarico.policies.linear import LinearPolicy

def test0():
    print()
    print('# test sequence labeler on mod data with DAgger')
    n_types = 10
    n_labels = 4

    data = [Example(x, y, n_labels) for x, y in macarico.util.make_sequence_mod_data(100, 5, n_types, n_labels)]

    # compute base features
    features = EmbeddingFeatures(n_types)
    #features = BOWFeatures(n_types)

    # optionally run RNN or CNN
    #features = RNN(features)
    features = DilatedCNN(features)

    # compute some attention
    attention = AttendAt(features, 'n') # or `lambda s: s.n`
    attention2 = FrontBackAttention(features)
    #attention = FrontBackAttention(features)
    #attention = SoftmaxAttention(features) # note: softmax doesn't work with BOWActor

    # build an actor
    #actor = RNNActor([attention, attention2], n_labels)
    actor = BOWActor([attention, attention2], n_labels)
    policy = LinearPolicy(actor, n_labels)
    
    #tRNN = Actor(
    #             [RNNFeatures(
    #                          n_types,
    #                          output_field = 'mytok_rnn')],
    #             [AttendAt(field='mytok_rnn')],
    #             n_labels)
    #policy = LinearPolicy(tRNN, n_labels)

    print(policy)
    
    learner = DAgger(HammingLossReference(), policy, ExponentialAnnealing(0.99))
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)

    macarico.util.trainloop(
        training_data   = data[:len(data)//2],
        dev_data        = data[len(data)//2:],
        policy          = policy,
        learner         = learner,
        losses          = HammingLoss(),
        optimizer       = optimizer,
        n_epochs        = 5,
        train_eval_skip = 1,
    )


if __name__ == '__main__':
    test0()
