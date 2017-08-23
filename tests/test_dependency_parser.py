from __future__ import division
import random
import sys
import torch
import testutil
testutil.reseed()

from macarico.annealing import ExponentialAnnealing, stochastic
from macarico.lts.maximum_likelihood import MaximumLikelihood
from macarico.lts.dagger import DAgger
from macarico.lts.aggrevate import AggreVaTe
from macarico.features.sequence import RNNFeatures, BOWFeatures
from macarico.features.actor import TransitionRNN, TransitionBOW
from macarico.policies.linear import LinearPolicy
from macarico.tasks.dependency_parser import DependencyAttention, Example, AttachmentLoss, AttachmentLossReference

import nlp_data

Features = RNNFeatures
#Features = BOWFeatures

Actor = TransitionRNN
#Actor = TransitionBOW


def test0():
    print
    print '# make sure dependency parser ref is one-step optimal'
    print
    tokens = 'the dinosaur ate a fly'.split()
    testutil.test_reference_on(AttachmentLossReference(),
                               AttachmentLoss,
                               Example(tokens,
                                       heads=[1, 2, 5, 4, 2],
                                       rels=None,
                                       n_rels=0),
                               verbose=False,
                               test_values=True)

    testutil.test_reference_on(AttachmentLossReference(),
                               AttachmentLoss,
                               Example(tokens,
                                       heads=[1, 2, 5, 4, 2],
                                       rels=[1, 2, 0, 1, 2],
                                       n_rels=3),
                               verbose=False,
                               test_values=True)

    print
    print '# testing on wsj'
    print
    train, _, _, _, _, _ = \
      nlp_data.read_wsj_deppar(labeled=False, n_tr=5000, n_de=0, n_te=0, max_length=40)

    random.shuffle(train)

    print 'number non-projective trees:', sum((x.is_non_projective() for x in train))
    train = train[:10]
    
    testutil.test_reference(AttachmentLossReference(),
                            AttachmentLoss,
                            train)

    # if sentence is A B #, and par(A) = par(B) = # = root
    # then want to
    #   s0: i=
    
    
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
    parse = parser.run_episode(AttachmentLossReference())
    print '## test rels=no'
    assert parse.heads == example.heads, 'got = %s, want = %s' % (parse.heads, example.heads)
    assert parse.rels == example.rels, 'got = %s, want = %s' % (parse.rels, example.rels)
    loss = AttachmentLoss().evaluate(example, parser)
    assert loss == 0, loss

    print '## test rels=yes'
    example = Example(tokens, heads=[1, 2, 5, 4, 2], rels=[1, 2, 0, 1, 2], n_rels=3)
    parser = example.mk_env()
    parse = parser.run_episode(AttachmentLossReference())
    assert parse.heads == example.heads, 'got = %s, want = %s' % (parse.heads, example.heads)
    assert parse.rels == example.rels, 'got = %s, want = %s' % (parse.rels, example.rels)
    loss = AttachmentLoss().evaluate(example, parser)
    assert loss == 0, loss


def test2(use_aggrevate=False):
    print
    print '# test simple branching trees, use_aggrevate=%s' % use_aggrevate

    # make simple branching trees
    T = 5
    n_types = 20
    data = []
    for _ in xrange(20):
        x = [random.randint(0,n_types-1) for _ in xrange(T)]
        y = [T for i in xrange(T)]
        #y = [0 if i > 0 else None for i in xrange(T)]
        data.append(Example(x, heads=y, rels=None, n_rels=0))

    tRNN = Actor([Features(n_types, output_field='tokens_feats')], [DependencyAttention()], 3)
    policy = LinearPolicy(tRNN, 3)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    p_rollin_ref = stochastic(ExponentialAnnealing(0.5))
    learner = (lambda: MaximumLikelihood(AttachmentLossReference(), policy)) \
              if not use_aggrevate else \
              (lambda: AggreVaTe(AttachmentLossReference(), policy, p_rollin_ref))
              

    testutil.trainloop(
        training_data   = data[:len(data)//2],
        dev_data        = data[len(data)//2:],
        policy          = policy,
        Learner         = learner,
        losses          = AttachmentLoss(),
        optimizer       = optimizer,
        train_eval_skip = 1,
        n_epochs        = 2,
    )


def test3(labeled=False, use_pos_stream=False, big_test=None, load_embeddings=None):
    # TODO: limit to short sentences
    print
    print '# Testing wsj parser, labeled=%s, use_pos_stream=%s, load_embeddings=%s' \
        % (labeled, use_pos_stream, load_embeddings)
    if big_test is None:
        train, dev, _, word_vocab, pos_vocab, relation_ids = \
          nlp_data.read_wsj_deppar(labeled=labeled, n_tr=50, n_de=50, n_te=0)
    else:
        train, dev, _, word_vocab, pos_vocab, relation_ids = \
          nlp_data.read_wsj_deppar(labeled=labeled)
        if big_test == 'medium':
            train = train[:200]
        elif big_test != 'big':
            train = train[:5000]

    initial_embeddings = None
    learn_embeddings = True
    d_emb = 50
    if load_embeddings is not None and load_embeddings != 'None':
        initial_embeddings = nlp_data.read_embeddings(load_embeddings, word_vocab)
        learn_embeddings = False
        d_emb = None
            
    n_actions = 3 + len(relation_ids)
    print '|word vocab| = %d, |pos vocab| = %d, n_actions = %d' % (len(word_vocab), len(pos_vocab), n_actions)

    # construct policy to learn    
    #inputs = [BOWFeatures(len(word_vocab), output_field='tokens_feats')]
    inputs = [RNNFeatures(len(word_vocab),
                          d_emb=d_emb,
                          initial_embeddings=initial_embeddings,
                          learn_embeddings=learn_embeddings,
                         )]
    foci = [DependencyAttention()]
    if use_pos_stream:
        inputs.append(RNNFeatures(len(pos_vocab),
                                  d_emb=10,
                                  d_rnn=10,
                                  input_field='pos',
                                  output_field='pos_rnn'))
        foci.append(DependencyAttention(field='pos_rnn'))

    policy = LinearPolicy(TransitionRNN(inputs, foci, n_actions), n_actions)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    p_rollin_ref  = stochastic(ExponentialAnnealing(0.9))

    def print_it():
        return
        print sum((p.norm().data[0] for p in policy.parameters()))

    print_it()
    # TODO: move this to a unit test.
    print 'reference loss on train = %g' % \
        testutil.evaluate(train, AttachmentLossReference(), AttachmentLoss())

    if big_test == 'predict':
        print 'stupid policy loss on train = %g' % \
            testutil.evaluate(train, AttachmentLossReference(), AttachmentLoss())
        return
    
    testutil.trainloop(
        training_data   = train,
        dev_data        = dev,
        policy          = policy,
        Learner         = lambda: DAgger(AttachmentLossReference(), policy, p_rollin_ref),
        losses          = AttachmentLoss(),
        optimizer       = optimizer,
        train_eval_skip = max(1, len(train) // 100),
        print_freq      = 25,
        n_epochs        = 4,
        run_per_epoch   = [p_rollin_ref.step, print_it],
    )

if __name__ == '__main__' and len(sys.argv) == 1:
    test0()
    test1()
    test2(False)
    test2(True)
    test3(False, False)
    test3(False, True )
    test3(True , False)
    test3(True , True )

if __name__ == '__main__' and len(sys.argv) >= 2:
    test3(False, True, sys.argv[1], sys.argv[2])
