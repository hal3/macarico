from __future__ import division, generators, print_function
import random
import torch
import torch.nn as nn
import numpy as np
import macarico.util
#macarico.util.reseed()
import sys

from macarico.lts.dagger import DAgger, Coaching
from macarico.lts.behavioral_cloning import BehavioralCloning
from macarico.lts.aggrevate import AggreVaTe
from macarico.lts.lols import LOLS, BanditLOLS
from macarico.lts.reinforce import Reinforce, LinearValueFn, A2C

from macarico.annealing import ExponentialAnnealing
from macarico.features.sequence import EmbeddingFeatures, BOWFeatures, RNN, DilatedCNN, AttendAt, FrontBackAttention, SoftmaxAttention, AverageAttention
from macarico.actors.rnn import RNNActor
from macarico.actors.bow import BOWActor
from macarico.policies.linear import LinearPolicy

import macarico.tasks.sequence_labeler as sl
import macarico.tasks.dependency_parser as dp

def build_learner(n_types, n_actions, ref, loss_fn, require_attention):
    # compute base features
    features = random.choice([lambda: EmbeddingFeatures(n_types),
                              lambda: BOWFeatures(n_types)])()

    # optionally run RNN or CNN
    features = random.choice([lambda: features,
                              lambda: RNN(features),
                              lambda: DilatedCNN(features)])()

    # maybe some nn magic
    if random.random() < 0.5:
        features = macarico.Torch(features,
                                  50, # final dimension, too hard to tell from list of layers :(
                                  [nn.Linear(features.dim, 50),
                                   nn.Tanh(),
                                   nn.Linear(50, 50),
                                   nn.Tanh()])

    # compute some attention
    if require_attention is not None:
        attention = [require_attention(features)]
    else:
        attention = [random.choice([lambda: AttendAt(features, 'n'), # or `lambda s: s.n`
                                    lambda: AverageAttention(features),
                                    lambda: FrontBackAttention(features),
                                    lambda: SoftmaxAttention(features)])()] # note: softmax doesn't work with BOWActor
        if random.random() < 0.2:
            attention.append(AttendAt(features, lambda s: s.N-s.n))

    # build an actor
    if any((isinstance(x, SoftmaxAttention) for x in attention)):
        actor = RNNActor(attention, n_actions)
    else:
        actor = random.choice([lambda: RNNActor(attention, n_actions),
                               lambda: BOWActor(attention, n_actions, history_length=3)])()

    # do something fun: add a torch module in the middle
    if random.random() < 0.5:
        actor = macarico.Torch(actor,
                               27, # final dimension, too hard to tell from list of layers :(
                               [nn.Linear(actor.dim, 27),
                                nn.Tanh()])

    # build the policy
    policy = LinearPolicy(actor, n_actions)
    parameters = policy.parameters()

    # build the learner
    if random.random() < 0.1: # A2C
        value_fn = LinearValueFn(actor)
        learner = A2C(policy, value_fn)
        parameters = list(parameters) + list(value_fn.parameters())
    else:
        learner = random.choice([BehavioralCloning(policy, ref),
                                 DAgger(policy, ref), #, ExponentialAnnealing(0.99))
                                 Coaching(policy, ref, policy_coeff=0.1),
                                 AggreVaTe(policy, ref),
                                 Reinforce(policy),
                                 BanditLOLS(policy, ref),
                                 LOLS(policy, ref, loss_fn)])
    
    return policy, learner, parameters

def test0(environment, n_epochs=1):
    n_examples = 2
    n_types = 10
    n_actions = 3

    if environment == 'sl':
        data = [sl.Example(x, y, n_actions) for x, y in macarico.util.make_sequence_mod_data(n_examples, 4, n_types, n_actions)]
        loss_fn = sl.HammingLoss()
        ref = sl.HammingLossReference()
        require_attention = None
    elif environment == 'dp':
        data = [dp.Example(tokens=[0, 1, 2, 3, 4],
                           heads= [1, 5, 4, 4, 1],
                           rels=None,
                           n_rels=0) for _ in range(n_examples)]
        loss_fn = dp.AttachmentLoss()
        ref = dp.AttachmentLossReference()
        require_attention = dp.DependencyAttention

    policy, learner, parameters = build_learner(n_types, n_actions, ref, loss_fn, require_attention)
    print(learner)

    optimizer = torch.optim.Adam(parameters, lr=0.001)

    macarico.util.trainloop(
        training_data   = data[:len(data)//2],
        dev_data        = data[len(data)//2:],
        policy          = policy,
        learner         = learner,
        losses          = loss_fn,
        optimizer       = optimizer,
        n_epochs        = n_epochs,
    )


if __name__ == '__main__':
    if len(sys.argv) == 1:
        seed = random.randint(0, 1e9)
    else:
        seed = int(sys.argv[1])
    print('seed', seed)
    macarico.util.reseed(seed)
    test0(environment=random.choice(['sl', 'dp']),
          n_epochs=1)
