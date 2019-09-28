import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
import sys
import macarico.util
macarico.util.reseed()

from macarico.annealing import ExponentialAnnealing, stochastic, EWMA
from macarico.lts.lols import BanditLOLS
from macarico.tasks.sequence_labeler import Example, HammingLoss, HammingLossReference
from macarico.features.sequence import RNNFeatures, AttendAt
from macarico.features.actor import TransitionRNN
from macarico.policies.bootstrap import BootstrapPolicy
from macarico.policies.linear import LinearPolicy

def test1(learning_method, exploration):
    print
    print '# testing learning_method=%d exploration=%d' % (learning_method, exploration)
    print
    n_types = 10
    n_labels = 2
    data = macarico.util.make_sequence_mod_data(100, 1, n_types, n_labels)
    data = [Example(x, y, n_labels) for x, y in data]

    bag_size = 5
    tRNN = [TransitionRNN([RNNFeatures(n_types)], [AttendAt()], n_labels) for i in range(bag_size)]
    policy = BootstrapPolicy(tRNN, n_labels)
    #policy = LinearPolicy(tRNN[0], n_labels)
    #print 'policy=', policy
    #print 'parameters=', list(policy.parameters())
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)

    p_rollin_ref  = stochastic(ExponentialAnnealing(0.9))
    p_rollout_ref = stochastic(ExponentialAnnealing(0.99999))

    macarico.util.trainloop(
        training_data   = data[:len(data)//2],
        dev_data        = data[len(data)//2:],
        policy          = policy,
        Learner         = lambda: BanditLOLS(HammingLossReference(),
                                             policy,
                                             p_rollin_ref,
                                             p_rollout_ref,
                                             learning_method,
                                             exploration,
                                             ),
        losses          = HammingLoss(),
        optimizer       = optimizer,
        run_per_batch   = [p_rollin_ref.step, p_rollout_ref.step],
        train_eval_skip = 1,
        n_epochs        = 2,
    )

if __name__ == '__main__':
    for learning_method in [BanditLOLS.LEARN_IPS, BanditLOLS.LEARN_DR, BanditLOLS.LEARN_MTR]:
        for exploration in [BanditLOLS.EXPLORE_BOLTZMANN, BanditLOLS.EXPLORE_BOOTSTRAP]:
            test1(learning_method, exploration)


