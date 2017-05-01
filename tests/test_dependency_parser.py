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

from test_sequence_labeler import evaluate
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
    loss = parser.loss_function(true_heads, true_rels)
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
        x = [random.randint(0,n_types-1) for _ in range(T)]
        y = [i+1 if i < 4 else None for i in range(T)]
        #y = [0 if i > 0 else None for i in range(T)]
        data.append((x,y))

    n_tr = len(data) // 2
    train = data[:n_tr]
    dev = data[n_tr:]

    policy = LinearPolicy(BiLSTMFeatures(DepParFoci(), n_types, 3), 3)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)

    for epoch in xrange(10):
        random.shuffle(train)
        for words, heads in train:
            parser = DependencyParser(words)
            loss = parser.loss_function(heads)
            learner = MaximumLikelihood(loss.reference, policy)
            optimizer.zero_grad()
            parser.run_episode(learner)
            learner.update(loss())
            optimizer.step()
        print 'error rate: tr %g de %g' % \
            (evaluate(DependencyParser, train, policy),
             evaluate(DependencyParser, dev, policy))

def test3():
    train,dev,test,word_vocab,pos_vocab,rel_id = nlp_data.read_wsj_deppar()
    #train = train[:2000]
    
    policy = LinearPolicy(BiLSTMFeatures(DepParFoci(),
                                         len(word_vocab),
                                         3,
                                         d_emb=500,
                                         n_layers=2,
                                         ),
                          3)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)

    if True:  # evaluate ref
        print 'reference loss on train = %g' % \
          evaluate(DependencyParser, ((w, h) for w, _, h, _ in train), None)

    def eval(epoch, ii):
        print >>sys.stderr, 'epoch %d.%d\t' % (epoch, ii),
        print >>sys.stderr, 'tr %g\t' % evaluate(DependencyParser, ((w,h) for w,_,h,_ in train[::20]), policy, False),
        print >>sys.stderr, 'de %g\t' % evaluate(DependencyParser, ((w,h) for w,_,h,_ in dev        ), policy, False),
        print >>sys.stderr, ''

    for epoch in xrange(200):
        random.shuffle(train)
        for ii, (words, _, heads, _) in enumerate(train):
            parser = DependencyParser(words)
            loss = parser.loss_function(heads)
            learner = MaximumLikelihood(loss.reference, policy)
            optimizer.zero_grad()
            res = parser.run_episode(learner)
            learner.update(loss())
            optimizer.step()
            if ii % max(1, len(train) // 100) == 0: eval(epoch, ii)

if __name__ == '__main__':
    #test1()
    #test2()
    test3()
