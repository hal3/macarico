from __future__ import division
import random
import sys
import torch

from macarico.annealing import stochastic, ExponentialAnnealing
from macarico.lts.reinforce import Reinforce
from macarico.lts.maximum_likelihood import MaximumLikelihood
from macarico.lts.dagger import DAgger
from macarico.lts.lols import BanditLOLS
from macarico.annealing import EWMA
from macarico.tasks.sequence_labeler import BiLSTMFeatures
from macarico.tasks.dependency_parser import ParseTree, DependencyParser, AttachmentLoss
from macarico import LinearPolicy

import testutil
import nlp_data

def test1():
    def random_policy(state):
        return random.choice(list(state.actions))

    # just test dependency structure without learning
    tokens = 'the dinosaur ate a fly'.split()
    print DependencyParser(tokens).run_episode(random_policy)
    print DependencyParser(tokens, n_rels=4).run_episode(random_policy)

    true_heads = [1, 2, 5, 4, 2]
    parser = DependencyParser(tokens)
    loss = parser.loss_function(true_heads)
    parse = parser.run_episode(loss.reference)
    print 'loss = %d, parse = %s' % (loss(), parse)

    true_rels = [1, 2, 0, 1, 2]
    parser = DependencyParser(tokens, n_rels=3)
    loss = parser.loss_function((true_heads, true_rels))
    parse = parser.run_episode(loss.reference)
    print 'loss = %d, parse = %s' % (loss(), parse)


class DepParFoci:
    arity = 2
    def __call__(self, state):
        buffer_pos = state.i if state.i < state.N else None
        stack_pos  = state.stack[-1] if state.stack else None
        #print '[foci=%s]' % [buffer_pos, stack_pos],
        return [buffer_pos, stack_pos]

def test2():
    # make simple branching trees
    T = 5
    n_types = 20
    data = []
    for _ in xrange(100):
        x = [random.randint(0,n_types-1) for _ in xrange(T)]
        y = [i+1 if i < 4 else None for i in xrange(T)]
        #y = [0 if i > 0 else None for i in xrange(T)]
        data.append((x,y))

    policy = LinearPolicy(BiLSTMFeatures(DepParFoci(), n_types, 3), 3)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)

    testutil.trainloop(
        Env             = DependencyParser,
        training_data   = data[:len(data)//2],
        dev_data        = data[len(data)//2:],
        policy          = policy,
        Learner         = lambda ref: MaximumLikelihood(ref, policy),
        optimizer       = optimizer,
        train_eval_skip = 1,
    )

def test3():
    train,dev,test,word_vocab,pos_vocab,rel_id = nlp_data.read_wsj_deppar()
    #train = train[:2000]

    # remove POS and relations from train/dev/test
    train = [(w,h) for ((w,_),(h,_)) in train]
    dev   = [(w,h) for ((w,_),(h,_)) in dev  ]
    test  = [(w,h) for ((w,_),(h,_)) in test ]

    # construct policy to learn
    policy = LinearPolicy(BiLSTMFeatures(DepParFoci(),
                                         len(word_vocab), 3,
                                         d_emb=500, n_layers=2), 3)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)

    print 'reference loss on train = %g' % \
        testutil.evaluate(DependencyParser, train, None)

    testutil.trainloop(
        Env             = DependencyParser,
        training_data   = train,
        dev_data        = dev,
        policy          = policy,
        Learner         = lambda ref: MaximumLikelihood(ref, policy),
        optimizer       = optimizer,
        train_eval_skip = 100,
        print_freq      = 100,
    )

if __name__ == '__main__':
    #test1()
    test2()
    test3()
