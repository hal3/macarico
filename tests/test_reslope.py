import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
import sys
import macarico.util
macarico.util.reseed()

from macarico.annealing import ExponentialAnnealing, stochastic
from macarico.lts.lols import BanditLOLS
from macarico.lts.reslope import Reslope
from macarico.lts.aggrevate import AggreVaTe
from macarico.tasks.sequence_labeler import Example, HammingLoss, HammingLossReference
from macarico.features.sequence import RNNFeatures, AttendAt
from macarico.features.actor import TransitionRNN
from macarico.policies.linear import LinearPolicy
from macarico.policies.bootstrap import BootstrapPolicy

def test1(use_bootstrap):
    n_types = 10
    n_labels = 4
    print
    print('# test sequence labeler on mod data with Reslope and', ('bootstrap' if use_bootstrap else 'boltzmann'), 'exploration')
    data = macarico.util.make_sequence_mod_data(3000, 6, n_types, n_labels)
    data = [Example(x, y, n_labels) for x, y in data]


    if not use_bootstrap:
        tRNN = TransitionRNN([RNNFeatures(n_types)], [AttendAt()], n_labels)
        policy = LinearPolicy(tRNN, n_labels)
    else:
        rnns = [TransitionRNN([RNNFeatures(n_types)], [AttendAt()], n_labels, h_name='h%d' % i)
                for i in range(5)]
        policy = BootstrapPolicy(rnns, n_labels)
        
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    
    p_ref  = stochastic(ExponentialAnnealing(0.9))

    macarico.util.trainloop(
        training_data   = data[:2048],
        dev_data        = data[2048:],
        policy          = policy,
        Learner         = lambda: Reslope(HammingLossReference(), policy, p_ref,
                                          exploration=BanditLOLS.EXPLORE_BOOTSTRAP if use_bootstrap else \
                                                      BanditLOLS.EXPLORE_BOLTZMANN
        ),
        losses          = HammingLoss(),
        optimizer       = optimizer,
        run_per_epoch   = [p_ref.step],
        train_eval_skip = 1,
        bandit_evaluation = True,
        n_epochs = 1,
    )

if __name__ == '__main__':
    test1(False)
    test1(True)
    
