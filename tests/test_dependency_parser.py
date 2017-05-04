from __future__ import division
import random
import torch

from macarico.annealing import stochastic, ExponentialAnnealing
from macarico.lts.reinforce import Reinforce
from macarico.lts.maximum_likelihood import MaximumLikelihood
from macarico.lts.dagger import DAgger
from macarico.lts.lols import BanditLOLS
from macarico.annealing import EWMA
from macarico.tasks.sequence_labeler import SequenceLabeling, RNNFeatures, TransitionRNN, SeqFoci, RevSeqFoci
from macarico.tasks.dependency_parser import ParseTree, DependencyParser, DepParFoci, Example
from macarico import LinearPolicy

import testutil
import nlp_data

def test1():
#    def random_policy(state):
#        return random.choice(list(state.actions))

    # just test dependency structure without learning
    tokens = 'the dinosaur ate a fly'.split()
#    print DependencyParser(tokens).run_episode(random_policy)
#    print DependencyParser(tokens, n_rels=4).run_episode(random_policy)

    example = Example(tokens, heads=[1, 2, 5, 4, 2], rels=None, n_rels=0)
    parser = example.mk_env()
    parse = parser.run_episode(parser.reference())
    print 'loss = %d, parse = %s' % (parser.loss(), parse)

    example = Example(tokens, heads=[1, 2, 0, 1, 2], rels=None, n_rels=0)
    parser = example.mk_env()
    parse = parser.run_episode(parser.reference())
    print 'loss = %d, parse = %s' % (parser.loss(), parse)


def test2():
    print '# test simple branching trees'

    # make simple branching trees
    T = 5
    n_types = 20
    data = []
    for _ in xrange(100):
        x = [random.randint(0,n_types-1) for _ in xrange(T)]
        y = [i+1 if i < 4 else None for i in xrange(T)]
        #y = [0 if i > 0 else None for i in xrange(T)]
        data.append(Example(x, heads=y, rels=None, n_rels=0))

    tRNN = TransitionRNN([RNNFeatures(n_types)], [DepParFoci()], 3)
    policy = LinearPolicy(tRNN, 3)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)

    testutil.trainloop(
        training_data   = data[:len(data)//2],
        dev_data        = data[len(data)//2:],
        policy          = policy,
        Learner         = lambda ref: MaximumLikelihood(ref, policy),
        optimizer       = optimizer,
        train_eval_skip = 1,
        n_epochs        = 2,
    )

def test3(use_pos_stream=False):
    print '# Testing wsj parser'
    train,dev,test,word_vocab,pos_vocab,rel_id = nlp_data.read_wsj_deppar()
    train = train[:200]
    dev = dev[:200]
    n_rels = len(rel_id)

    print 'n_rels = %s' % n_rels
    #print 'rels = %s' % rel_id

    # construct policy to learn
    inputs = [RNNFeatures(len(word_vocab))]
    foci = [DepParFoci()]
    if use_pos_stream:
        inputs.append(RNNFeatures(len(pos_vocab),
                                  d_emb=10,
                                  d_rnn=10,
                                  input_field='pos',
                                  output_field='pos_rnn'))
        foci.append(DepParFoci(field='pos_rnn'))

    policy = LinearPolicy(TransitionRNN(inputs, foci, 3), 3)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)

    # TODO: move this to a unit test.
    print 'reference loss on train = %g' % \
        testutil.evaluate(train, lambda s: s.reference()(s))

    testutil.trainloop(
        training_data   = train,
        dev_data        = dev,
        policy          = policy,
        Learner         = lambda ref: MaximumLikelihood(ref, policy),
        optimizer       = optimizer,
        train_eval_skip = 100,
        print_freq      = 50,
        n_epochs        = 1,
    )

if __name__ == '__main__':
#    test1()
#    test2()
    test3(False)
    test3(True)
