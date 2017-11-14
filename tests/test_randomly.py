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

from macarico.annealing import ExponentialAnnealing, NoAnnealing, Averaging, EWMA
from macarico.features.sequence import EmbeddingFeatures, BOWFeatures, RNN, DilatedCNN, AttendAt, FrontBackAttention, SoftmaxAttention, AverageAttention
from macarico.actors.rnn import RNNActor
from macarico.actors.bow import BOWActor
from macarico.policies.linear import LinearPolicy

import macarico.tasks.sequence_labeler as sl
import macarico.tasks.dependency_parser as dp
import macarico.tasks.seq2seq as s2s
import macarico.tasks.pocman as pocman
import macarico.tasks.cartpole as cartpole
import macarico.tasks.blackjack as blackjack
import macarico.tasks.hexgame as hexgame
import macarico.tasks.gridworld as gridworld
import macarico.tasks.pendulum as pendulum
import macarico.tasks.mdp as mdp
import macarico.tasks.mountain_car as car

def build_learner(n_types, n_actions, ref, loss_fn, require_attention):
    dim=50
    features = RNN(EmbeddingFeatures(n_types, d_emb=dim), d_rnn=dim)
    attention = require_attention or AttendAt
    attention = attention(features)
    actor = RNNActor([attention], n_actions)
    policy = LinearPolicy(actor, n_actions)
    #learner = LOLS(policy, ref, loss_fn, p_rollin_ref=NoAnnealing(1))
    learner = AggreVaTe(policy, ref, p_rollin_ref=ExponentialAnnealing(0.9))
    return policy, learner, policy.parameters()

def build_random_learner(n_types, n_actions, ref, loss_fn, require_attention):
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

def test_rl(environment, n_epochs=10000):
    tasks = {
        'pocman': (pocman.MicroPOCMAN, pocman.LocalPOCFeatures, pocman.POCLoss, pocman.POCReference),
        'cartpole': (cartpole.CartPoleEnv, cartpole.CartPoleFeatures, cartpole.CartPoleLoss, None),
        'blackjack': (blackjack.Blackjack, blackjack.BlackjackFeatures, blackjack.BlackjackLoss, None),
        'hex': (hexgame.Hex, hexgame.HexFeatures, hexgame.HexLoss, None),
        'gridworld': (gridworld.make_default_gridworld, gridworld.LocalGridFeatures, gridworld.GridLoss, None),
        'pendulum': (pendulum.Pendulum, pendulum.PendulumFeatures, pendulum.PendulumLoss, None),
        'car': (car.MountainCar, car.MountainCarFeatures, car.MountainCarLoss, None),
        'mdp': (lambda: mdp.make_ross_mdp()[0], lambda: mdp.MDPFeatures(3), mdp.MDPLoss, lambda: mdp.make_ross_mdp()[1]),
    }
              
    mk_ex, mk_fts, loss_fn, ref = tasks[environment]
    ex = mk_ex()

    features = mk_fts()
    attention = AttendAt(features, position=lambda _: 0)
    actor = BOWActor([attention], ex.n_actions)
    policy = LinearPolicy(actor, ex.n_actions)
    learner = Reinforce(policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    losses, objs = [], []
    for epoch in range(1, 1+n_epochs):
        optimizer.zero_grad()
        env = ex.mk_env()
        env.run_episode(learner)
        loss_val = loss_fn()(ex, env)
        obj = learner.update(loss_val)
        optimizer.step()
        losses.append(loss_val)
        objs.append(obj)
        #losses.append(loss)
        if epoch%100 == 0:
            print(epoch, np.mean(losses[-500:]), np.mean(objs[-500:]))
    

def test_sp(environment, n_epochs=1, n_examples=1, fixed=False):
    n_types = 100 if fixed else 10
    length = 5 if fixed else 4
    n_actions = 5 if fixed else 3

    if environment == 'sl':
        data = [sl.Example(x, y, n_actions) for x, y in macarico.util.make_sequence_mod_data(n_examples, length, n_types, n_actions)]
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
    elif environment == 's2s':
        data = [s2s.Example(x, [int(i+1) for i in y], n_actions) \
                for x, y in macarico.util.make_sequence_mod_data(n_examples, length, n_types-1, n_actions-1)]
        loss_fn = s2s.EditDistance()
        ref = s2s.EditDistanceReference()
        require_attention = AttendAt# SoftmaxAttention

    builder = build_learner if fixed else build_random_learner
    policy, learner, parameters = builder(n_types, n_actions, ref, loss_fn, require_attention)
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
    macarico.util.reseed(20001)
    test_rl(sys.argv[1])
"""    
    fixed = False
    if len(sys.argv) == 1:
        seed = random.randint(0, 1e9)
    elif sys.argv[1] == 'fixed':
        seed = 90210
        fixed = True
    else:
        seed = int(sys.argv[1])
    print('seed', seed)
    macarico.util.reseed(seed)
    test_sp(environment='s2s' if fixed else random.choice(['sl', 'dp', 's2s']),
            n_epochs=1,
            n_examples=1024, # if fixed else 2,
            fixed=fixed)
"""