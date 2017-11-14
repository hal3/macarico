from __future__ import division, generators, print_function
import random
import torch
import torch.nn as nn
import numpy as np
import macarico.util
macarico.util.reseed()

from macarico.lts.dagger import DAgger, Coaching
from macarico.lts.behavioral_cloning import BehavioralCloning
from macarico.lts.aggrevate import AggreVaTe
from macarico.lts.lols import LOLS, BanditLOLS
from macarico.lts.reinforce import Reinforce, LinearValueFn, A2C

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
    length = 3

    data = [Example(x, y, n_labels) for x, y in macarico.util.make_sequence_mod_data(100, length, n_types, n_labels)]

    # compute base features
    features = EmbeddingFeatures(n_types)
    #features = BOWFeatures(n_types)

    # optionally run RNN or CNN
    features = RNN(features)
    #features = DilatedCNN(features)
    # add some nn magic
    #features = macarico.Torch(features,
    #                          50, # final dimension, too hard to tell from list of layers :(
    #                          [nn.Linear(features.dim, 50),
    #                           nn.Tanh(),
    #                           nn.Linear(50, 50),
    #                           nn.Tanh()])
                              

    # compute some attention
    attention = AttendAt(features, 'n') # or `lambda s: s.n`
    #attention2 = FrontBackAttention(features)
    #attention = FrontBackAttention(features)
    #attention = SoftmaxAttention(features) # note: softmax doesn't work with BOWActor

    # build an actor
    actor = RNNActor([attention], n_labels)
    #actor = BOWActor([attention, attention2], n_labels, history_length=3)

    # do something fun: add a torch module in the middle
    #actor = macarico.Torch(actor,
    #                       27, # final dimension, too hard to tell from list of layers :(
    #                       [nn.Linear(actor.dim, 27),
    #                        nn.Tanh()])
    
    policy = LinearPolicy(actor, n_labels)

    print(policy)

    ref = HammingLossReference()
    #learner = BehavioralCloning(policy, ref)
    #learner = DAgger(policy, ref) #, ExponentialAnnealing(0.99))
    #learner = Coaching(policy, ref, policy_coeff=0.1)
    #learner = AggreVaTe(policy, ref)
    #learner = Reinforce(policy)
    #value_fn = LinearValueFn(actor)
    #learner = A2C(policy, value_fn)
    #learner = BanditLOLS(policy, ref)
    learner = LOLS(policy, ref, HammingLoss())

    parameters = policy.parameters()
    #parameters = list(policy.parameters()) + list(value_fn.parameters()) # for A2C
    optimizer = torch.optim.Adam(parameters, lr=0.001)

    macarico.util.trainloop(
        training_data   = data[:len(data)//2],
        dev_data        = data[len(data)//2:],
        policy          = policy,
        learner         = learner,
        losses          = HammingLoss(),
        optimizer       = optimizer,
        n_epochs        = 10,
        train_eval_skip = 1,
    )


if __name__ == '__main__':
    test0()
