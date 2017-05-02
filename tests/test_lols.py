from __future__ import division
import numpy as np
import random
import torch
import sys

from macarico.annealing import ExponentialAnnealing, stochastic
import macarico.lts.lols as LOLS
from macarico.tasks.sequence_labeler import SequenceLabeling, BiLSTMFeatures, SeqFoci
from macarico import LinearPolicy

from testutil import evaluate

def test1():
    # sample data
    n_types = 10
    n_labels = 4
    N = 20
    T = 6
    data = []
    for _ in xrange(N):
        x = np.random.randint(n_types, size=T)
        y = (x+1) % n_labels
        data.append((x,y))
    train, dev = data[:N//2], data[N//2:]
    #print train

    # run
    
    policy = LinearPolicy(BiLSTMFeatures(SeqFoci(), n_types,
                                         n_labels, d_emb=10, d_actemb=4), n_labels)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    
    p_rollin_ref  = ExponentialAnnealing(0.9)
    p_rollout_ref = ExponentialAnnealing(0.9)

    for epoch in xrange(100):
        for words, labels in train:
            env = lambda: SequenceLabeling(words, n_labels)
            optimizer.zero_grad()
            LOLS.lols(env, labels, policy, p_rollin_ref, p_rollout_ref)
            optimizer.step()

        print 'error rate: train %g dev %g' % \
            (evaluate(lambda w: SequenceLabeling(w, n_labels), train, policy),
             evaluate(lambda w: SequenceLabeling(w, n_labels), dev, policy))

if __name__ == '__main__':
    test1()
    
