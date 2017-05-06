from __future__ import division
import random
import torch
import testutil
testutil.reseed()

from macarico.lts.maximum_likelihood import MaximumLikelihood
from macarico.features.sequence import RNNFeatures
from macarico.features.actor import TransitionRNN
from macarico.policies.linear import LinearPolicy
from macarico.tasks.dependency_parser import DepParFoci, Example

import nlp_data

def test1():
    print
    print '# test dependency structure without learning'

    tokens = 'the dinosaur ate a fly'.split()

    print '## test random policy'
    example = Example(tokens, heads=None, rels=None, n_rels=0)
    print '### rels=no' 
    print example.mk_env().run_episode(lambda s: random.choice(list(s.actions)))
    print '### rels=yes'
    example = Example(tokens, heads=None, rels=None, n_rels=4)
    print example.mk_env().run_episode(lambda s: random.choice(list(s.actions)))

    example = Example(tokens, heads=[1, 2, 5, 4, 2], rels=None, n_rels=0)
    parser = example.mk_env()
    parse = parser.run_episode(parser.reference())
    print '## test rels=no'
    assert parse.heads == example.heads, 'got = %s, want = %s' % (parse.heads, example.heads)
    assert parse.rels == example.rels, 'got = %s, want = %s' % (parse.rels, example.rels)
    assert parser.loss() == 0, parser.loss()

    print '## test rels=yes'
    example = Example(tokens, heads=[1, 2, 5, 4, 2], rels=[1, 2, 0, 1, 2], n_rels=3)
    parser = example.mk_env()
    parse = parser.run_episode(parser.reference())
    assert parse.heads == example.heads, 'got = %s, want = %s' % (parse.heads, example.heads)
    assert parse.rels == example.rels, 'got = %s, want = %s' % (parse.rels, example.rels)
    assert parser.loss() == 0, parser.loss()


def test2():
    print
    print '# test simple branching trees'

    # make simple branching trees
    T = 5
    n_types = 20
    data = []
    for _ in xrange(20):
        x = [random.randint(0,n_types-1) for _ in xrange(T)]
        y = [i+1 for i in xrange(T)]
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


def test3(labeled=False, use_pos_stream=False):
    print
    print '# Testing wsj parser, labeled=%s, use_pos_stream=%s' % (labeled, use_pos_stream)
    train, dev, _, word_vocab, pos_vocab, relation_ids = \
      nlp_data.read_wsj_deppar(labeled=labeled, n_tr=50, n_de=50, n_te=0)

    print '|word vocab| = %d, |pos vocab| = %d' % (len(word_vocab), len(pos_vocab))
    n_actions = 3 + len(relation_ids)

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

    policy = LinearPolicy(TransitionRNN(inputs, foci, n_actions), n_actions)
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
        train_eval_skip = 25,
        print_freq      = 25,
        n_epochs        = 1,
    )

if __name__ == '__main__':
    test1()
    test2()
    test3(False, False)
    test3(False, True )
    test3(True , False)
    test3(True , True )

