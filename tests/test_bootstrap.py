from __future__ import division
import numpy as np
import random
import dynet as dy
import sys
import macarico.util
macarico.util.reseed()

from macarico.annealing import ExponentialAnnealing, stochastic, EWMA
from macarico.lts.lols import BanditLOLS
from macarico.tasks.sequence_labeler import Example, HammingLoss, HammingLossReference
from macarico.features.sequence import RNNFeatures, AttendAt
from macarico.features.actor import TransitionRNN
from macarico.policies.linear import LinearPolicy

def test1(learning_method, exploration):
    print
    print '# testing learning_method=%d exploration=%d' % (learning_method, exploration)
    print
    n_types = 10
    n_labels = 4
    data = macarico.util.make_sequence_mod_data(100, 6, n_types, n_labels)
    data = [Example(x, y, n_labels) for x, y in data]

    dy_model = dy.ParameterCollection()
    tRNN = TransitionRNN(dy_model, [RNNFeatures(dy_model, n_types)], [AttendAt()], n_labels)
    policy = LinearPolicy(dy_model, tRNN, n_labels)
    optimizer = dy.AdamTrainer(dy_model, alpha=0.001)

    p_rollin_ref  = stochastic(ExponentialAnnealing(0.9))
    p_rollout_ref = stochastic(ExponentialAnnealing(0.99999))
    baseline = EWMA(0)

    macarico.util.trainloop(
        training_data   = data[:len(data)//2],
        dev_data        = data[len(data)//2:],
        policy          = policy,
        Learner         = lambda: BanditLOLS(HammingLossReference(),
                                             policy,
                                             p_rollin_ref,
                                             p_rollout_ref,
                                             learning_method,  # LEARN_IPS, LEARN_DR, LEARN_BIASED
                                             exploration,
                                             baseline=baseline,
                                             ),
        losses          = HammingLoss(),
        optimizer       = optimizer,
        run_per_epoch   = [p_rollin_ref.step, p_rollout_ref.step],
        train_eval_skip = 10,
    )

if __name__ == '__main__':
    for learning_method in [BanditLOLS.LEARN_BIASED, BanditLOLS.LEARN_IPS, BanditLOLS.LEARN_DR]:
        for exploration in [BanditLOLS.EXPLORE_UNIFORM, BanditLOLS.EXPLORE_BOLTZMANN, BanditLOLS.EXPLORE_BOLTZMANN_BIASED]:
            test1(learning_method, exploration)


#import dynet as dy
#from macarico.policies.bootstrap import build_policy_bag
#from macarico.policies.bootstrap import BootstrapPolicy
#
#if __name__ == '__main__':
#    num_actions = 10
#    bag_size = 16
#    policy_bag = []
#    state = None
#    dy_model = dy.ParameterCollection()
#    features_bag = [None for i in range(bag_size)]
#    loss_fn = 'squared'
#    print(build_policy_bag(dy_model, features_bag, num_actions, loss_fn))
#    bootstrap_policy = BootstrapPolicy(dy_model, features_bag, num_actions, loss_fn)
##    print(bootstrap_probabilities(num_actions, bag_size, policy_bag, state))
