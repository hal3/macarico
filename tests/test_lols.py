from __future__ import division
import numpy as np
import random
import torch
import sys

from macarico.annealing import ExponentialAnnealing, stochastic
import macarico.lts.lols as LOLS
from macarico.tasks.sequence_labeler import SequenceLabeling, BiLSTMFeatures, SeqFoci
from macarico import LinearPolicy

def test1():
    # sample data
    n_types = 10
    n_labels = 4
    data = []
    for _ in xrange(50):
        x = np.random.randint(n_types, size=6)
        y = (x+1) % n_labels
        data.append((x,y))
    train, dev = data[:25], data[25:]

    # run
    
    policy = LinearPolicy(BiLSTMFeatures(SeqFoci(), n_types, n_labels), n_labels)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    
    p_rollin_ref  = ExponentialAnnealing(0.5)
    p_rollout_ref = ExponentialAnnealing(0.5)

    for epoch in xrange(10):
        for words, labels in train:
            env = SequenceLabeling(words, n_labels)
            loss = env.loss_function(labels)
            optimizer.zero_grad()
            LOLS.lols(env, policy, loss.reference, p_rollin_ref, p_rollout_ref)
            optimizer.step()

        print 'error rate: train %g dev %g' % \
            (evaluate(Env, train, policy),
             evaluate(Env, dev, policy))

if __name__ == '__main__':
    test1()
    
