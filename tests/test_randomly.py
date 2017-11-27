from __future__ import division, generators, print_function
import random
import torch
import torch.nn as nn
import numpy as np
import macarico.util
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
from macarico.policies.linear import *

import macarico.tasks.sequence_labeler as sl
import macarico.tasks.dependency_parser as dep
import macarico.tasks.seq2seq as s2s
import macarico.tasks.pocman as pocman
import macarico.tasks.cartpole as cartpole
import macarico.tasks.blackjack as blackjack
import macarico.tasks.hexgame as hexgame
import macarico.tasks.gridworld as gridworld
import macarico.tasks.pendulum as pendulum
import macarico.tasks.mdp as mdp
import macarico.tasks.mountain_car as car

def debug_on_assertion(type, value, tb):
   if hasattr(sys, 'ps1') or not sys.stderr.isatty() or type != AssertionError:
      sys.__excepthook__(type, value, tb)
   else:
      import traceback, ipdb
      traceback.print_exception(type, value, tb)
      print()
      ipdb.pm()
sys.excepthook = debug_on_assertion

def build_learner(n_types, n_actions, ref, loss_fn, require_attention):
    #features = RNN(EmbeddingFeatures(n_types, d_emb=dim), d_rnn=dim)
    features = BOWFeatures(n_types)
    attention = require_attention or AttendAt
    attention = attention(features)
    actor = BOWActor([attention], n_actions)
    policy = WMCPolicy(actor, n_actions)
    learner = BehavioralCloning(policy, ref)
    #learner = LOLS(policy, ref, loss_fn())
    #learner = Reinforce(policy)
    #value_fn = LinearValueFn(actor)
    #learner = A2C(policy, value_fn)
    #LOLS, BanditLOLS, Reinforce, A2C
    return policy, learner, list(policy.parameters()) #+ list(value_fn.parameters())

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
        actor = random.choice([lambda: RNNActor(attention,
                                                n_actions,
                                                d_actemb=random.choice([None,5]),
                                                cell_type=random.choice(['RNN', 'GRU', 'LSTM'])),
                               lambda: BOWActor(attention, n_actions, act_history_length=3, obs_history_length=2)])()

    # do something fun: add a torch module in the middle
    if random.random() < 0.5:
        actor = macarico.Torch(actor,
                               27, # final dimension, too hard to tell from list of layers :(
                               [nn.Linear(actor.dim, 27),
                                nn.Tanh()])

    # build the policy
    policy = random.choice([lambda: CSOAAPolicy(actor, n_actions, 'huber'),
                            lambda: CSOAAPolicy(actor, n_actions, 'squared'),
                            lambda: WMCPolicy(actor, n_actions, 'huber'),
                            lambda: WMCPolicy(actor, n_actions, 'hinge'),
                            lambda: WMCPolicy(actor, n_actions, 'multinomial'),
                           ])()
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
    print('rl', environment)
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
    policy = CSOAAPolicy(actor, ex.n_actions)
    learner = Reinforce(policy)
    print(learner)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    losses, objs = [], []
    for epoch in range(1, 1+n_epochs):
        optimizer.zero_grad()
        env = ex.mk_env()
        env.run_episode(learner)
        loss_val = loss_fn()(ex, env)
        obj = learner.get_objective(loss_val)
        if not isinstance(obj, float):
            obj.backward()
            optimizer.step()
            obj = obj.data[0]
        losses.append(loss_val)
        objs.append(obj)
        #losses.append(loss)
        if epoch%100 == 0 or epoch==n_epochs:
            print(epoch, np.mean(losses[-500:]), np.mean(objs[-500:]))
    

def test_sp(environment, n_epochs=1, n_examples=4, fixed=False, gpu_id=None):
    print('sp', environment)
    n_types = 50 if fixed else 10
    length = 6 if fixed else 4
    n_actions = 9 if fixed else 3

    if environment == 'sl':
        data = [sl.Example(x, y, n_actions) \
                for x, y in macarico.util.make_sequence_mod_data(n_examples, length, n_types, n_actions)]
        loss_fn = sl.HammingLoss
        ref = sl.HammingLossReference()
        require_attention = None
    elif environment == 'dp':
        data = [dep.Example(tokens=[0, 1, 2, 3, 4],
                           heads= [1, 5, 4, 4, 1],
                           rels=None,
                           n_rels=0) for _ in range(n_examples)]
        loss_fn = dep.AttachmentLoss
        ref = dep.AttachmentLossReference()
        require_attention = dep.DependencyAttention
    elif environment == 's2s':
        data = [s2s.Example(x, [int(i+1) for i in y], n_actions) \
                for x, y in macarico.util.make_sequence_mod_data(n_examples, length, n_types-1, n_actions-1)]
        loss_fn = s2s.EditDistance
        ref = s2s.NgramFollower()
        require_attention = AttendAt# SoftmaxAttention

    builder = build_learner if fixed else build_random_learner

    while True:
        policy, learner, parameters = builder(n_types, n_actions, ref, loss_fn, require_attention)
        if fixed or not (environment == 's2s' and (isinstance(learner, AggreVaTe) or isinstance(learner, Coaching))):
            break
            
    print(learner)

    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        policy = policy.cuda()
        # TODO do we need to .cuda() everything else? maybe this should go in trainloop so it's isolated?
        # TODO call .cpu() on everything when we need it on the cpu
        # TODO replace:
        #   torch.zeros(...) -> self._new(...).zero_()
        #   torch.LongTensor(...) -> self._new(...).long()
        #   onehot -> onehot(new)
    
    optimizer = torch.optim.Adam(parameters, lr=0.1)

    macarico.util.trainloop(
        training_data   = data[len(data)//2:],
        dev_data        = data[:len(data)//2],
        policy          = policy,
        learner         = learner,
        losses          = [loss_fn, loss_fn, loss_fn],
        optimizer       = optimizer,
        n_epochs        = n_epochs,
        progress_bar    = fixed,
        minibatch_size  = 1, #random.choice([1,2]),
    )

#if __name__ == '__main__':
#    macarico.util.reseed(20001)
#    test_rl(sys.argv[1])
    
if __name__ == '__main__':
    gpu_id = None # run on CPU
    fixed = False
    if len(sys.argv) == 1:
        seed = random.randint(0, 1e9)
    elif sys.argv[1] == 'fixed':
        seed = 90210
        fixed = True
    else:
        seed = int(sys.argv[1])
    print('seed', seed)
    macarico.util.reseed(seed, gpu_id=gpu_id)
    if fixed or np.random.random() < 0.8:
        test_sp(environment='sl' if fixed else random.choice(['sl', 'dp', 's2s']),
                n_epochs=1,
                n_examples=2**12 if fixed else 4,
                fixed=fixed,
                gpu_id=gpu_id)
    else:
        test_rl(random.choice('pocman cartpole blackjack hex gridworld pendulum car mdp'.split()),
                n_epochs=10)
    

