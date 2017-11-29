from __future__ import division, generators, print_function
import random
import sys
#import torch
import macarico.util
macarico.util.reseed()

#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#from torch.autograd import Variable as Var
from macarico.data import nlp_data
from macarico.data.types import Dependencies
from macarico.lts.dagger import DAgger, Coaching
from macarico.lts.behavioral_cloning import BehavioralCloning
from macarico.lts.aggrevate import AggreVaTe
from macarico.lts.lols import LOLS, BanditLOLS
from macarico.lts.reinforce import Reinforce, LinearValueFn, A2C

from macarico.tasks.dependency_parser import DependencyParser, AttachmentLossReference, AttachmentLoss, DependencyAttention, GlobalAttachmentLoss
from macarico.annealing import ExponentialAnnealing, NoAnnealing, Averaging, EWMA
from macarico.features.sequence import EmbeddingFeatures, BOWFeatures, RNN, DilatedCNN, AttendAt, FrontBackAttention, SoftmaxAttention, AverageAttention
from macarico.actors.rnn import RNNActor
from macarico.actors.bow import BOWActor
from macarico.policies.linear import *
from macarico import util

def test0():
    print()
    print('# make sure dependency parser ref is one-step optimal')
    print()
    tokens = 'the dinosaur ate a fly'.split()
    util.test_reference_on(DependencyParser,
                           AttachmentLossReference(),
                           AttachmentLoss,
                           Dependencies(tokens,
                                  heads=[1, 2, 5, 4, 2],
                                  token_vocab=5),
                           verbose=False,
                           test_values=True)

    util.test_reference_on(DependencyParser,
                           AttachmentLossReference(),
                           AttachmentLoss,
                           Dependencies(tokens,
                                  heads=[1, 2, 5, 4, 2],
                                  rels=[1, 2, 0, 1, 2],
                                  token_vocab=5,
                                  rel_vocab=3),
                           verbose=False,
                           test_values=True)

    print()
    print('# testing on wsj')
    print()
    train, _, _, _, _, _ = \
      nlp_data.read_wsj_deppar(labeled=False, n_tr=5000, n_de=0, n_te=0, max_length=40)

    random.shuffle(train)

    print('number non-projective trees:', sum((not x.is_projective() for x in train)))

    train = [x for x in train if x.is_projective()]
    #train = sorted(train, key=lambda x: len(x.tokens))
    train = train[:10]
    #train = [train[218]]
    
    util.test_reference(DependencyParser,
                        AttachmentLossReference(),
                        AttachmentLoss,
                        train,
                        test_values=True,
                        )

    # if sentence is A B #, and par(A) = par(B) = # = root
    # then want to
    #   s0: i=
    
    
def test1():
    print()
    print('# test dependency structure without learning')

    tokens = 'the dinosaur ate a fly'.split()

    print('## test random policy')
    example = Example(tokens, heads=None, rels=None, n_rels=0)
    print('### rels=no' )
    print(example.mk_env().run_episode(lambda s: random.choice(list(s.actions))))
    print('### rels=yes')
    example = Example(tokens, heads=None, rels=None, n_rels=4)
    print(example.mk_env().run_episode(lambda s: random.choice(list(s.actions))))

    example = Example(tokens, heads=[1, 2, 5, 4, 2], rels=None, n_rels=0)
    parser = example.mk_env()
    parse = parser.run_episode(AttachmentLossReference())
    print('## test rels=no')
    assert parse.heads == example.heads, 'got = %s, want = %s' % (parse.heads, example.heads)
    assert parse.rels == example.rels, 'got = %s, want = %s' % (parse.rels, example.rels)
    loss = AttachmentLoss().evaluate(example, parser)
    assert loss == 0, loss

    print('## test rels=yes')
    example = Example(tokens, heads=[1, 2, 5, 4, 2], rels=[1, 2, 0, 1, 2], n_rels=3)
    parser = example.mk_env()
    parse = parser.run_episode(AttachmentLossReference())
    assert parse.heads == example.heads, 'got = %s, want = %s' % (parse.heads, example.heads)
    assert parse.rels == example.rels, 'got = %s, want = %s' % (parse.rels, example.rels)
    loss = AttachmentLoss().evaluate(example, parser)
    assert loss == 0, loss


def test2(use_aggrevate=False):
    print()
    print('# test simple branching trees, use_aggrevate=%s' % use_aggrevate)

    # make simple branching trees
    T = 5
    n_types = 20
    data = []
    for _ in range(20):
        x = [random.randint(0,n_types-1) for _ in range(T)]
        y = [T for i in range(T)]
        #y = [0 if i > 0 else None for i in range(T)]
        data.append(Example(x, heads=y, rels=None, n_rels=0))

    actor = RNNActor([DependencyAttention(RNN(EmbeddingFeatures(n_types)))], 3)
    policy = CSOAAPolicy(actor, actor.n_actions)
    learner = (BehavioralCloning if not use_aggrevate else AggreVaTe)(policy, AttachmentLossReference())
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)

    util.TrainLoop(DependencyParser, policy, learner, optimizer,
                   losses=AttachmentLoss,
                   progress_bar=False,
                   minibatch_size=1
    ).train(data[0:10],#:len(data)//2],
            None,
            n_epochs=10)


def test3(labeled=False, use_tag_stream=False, big_test=None, load_embeddings=None):
    # TODO: limit to short sentences
    print()
    print('# Testing wsj parser, labeled=%s, use_tag_stream=%s, load_embeddings=%s' \
        % (labeled, use_tag_stream, load_embeddings))
    if big_test is None:
        train, dev, _, word_vocab, tag_vocab, rel_vocab = \
          nlp_data.read_wsj_deppar(labeled=labeled, n_tr=50, n_de=50, n_te=0)
    else:
        train, dev, _, word_vocab, tag_vocab, rel_vocab = \
          nlp_data.read_wsj_deppar(labeled=labeled, min_freq=2)
        if big_test == 'medium':
            train = train[:200]
        elif big_test != 'big':
            train = train[:1000]

    initial_embeddings = None
    learn_embeddings = True
    d_emb, d_rnn, d_actor = 256, 256, 256
    if load_embeddings is not None and load_embeddings != 'None':
        learn_embeddings = True
        if load_embeddings[0] == '!':
            learn_embeddings = False
            load_embeddings = load_embeddings[1:]
        initial_embeddings = nlp_data.read_embeddings(load_embeddings, word_vocab)
            
    n_actions = 3 + len(rel_vocab or [])
    print('|word vocab| = %d, |tag vocab| = %d, n_actions = %d' % (len(word_vocab), len(tag_vocab), n_actions))

    # construct policy to learn    
    word_embed = EmbeddingFeatures(len(word_vocab),
                                   d_emb=d_emb if initial_embeddings is None else None,
                                   initial_embeddings=initial_embeddings,
                                   learn_embeddings=learn_embeddings)
    word_features = RNN(word_embed, d_rnn)
    #word_features = DilatedCNN(word_embed)
    attention = [DependencyAttention(word_features)]
    if use_tag_stream:
        tag_features = RNN(BOWFeatures(len(tag_vocab), input_field='tags'), d_rnn=10)
        #tag_features = DilatedCNN(BOWFeatures(len(tag_vocab), input_field='tags'))
        attention.append(DependencyAttention(tag_features))
    
    
    #actor = BOWActor(attention, n_actions)
    actor = RNNActor(attention, n_actions, d_hid=d_actor)
    policy = CSOAAPolicy(actor, n_actions)
    
    learner = DAgger(policy, AttachmentLossReference(), p_rollin_ref=ExponentialAnnealing(0.99999))
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)

    def print_it():
        return
        print(sum((p.norm().data[0] for p in policy.parameters())))

    print_it()
    # TODO: move this to a unit test.
    print('reference loss on train = %g' % \
        util.evaluate(DependencyParser, train, AttachmentLossReference(), AttachmentLoss()))

    if big_test == 'predict':
        print('stupid policy loss on train = %g' % \
            util.evaluate(DependencyParser, train, AttachmentLossReference(), AttachmentLoss()))
        return
    
    util.TrainLoop(DependencyParser, policy, learner, optimizer,
                   losses = [AttachmentLoss, GlobalAttachmentLoss],
                   progress_bar = True,
                   minibatch_size = 1
    ).train(train, dev, 10) # TODO fix bug in progress_bar when n_tr > print_freq

def get_transition_sequences(fname):
    tr, _, _, _, _, _ = nlp_data.read_wsj_deppar(fname, n_tr=99999999, n_de=0, n_te=0)
    tr = [ex for ex in tr if ex.is_projective()]
    util.test_reference(
        AttachmentLossReference(),
        AttachmentLoss,
        tr,
        verbose=True,
        except_on_failure=False)

#get_transition_sequences(sys.argv[1])
#sys.exit(0)
    
if __name__ == '__main__' and len(sys.argv) == 1:
    test0()
    #test1()
    #test2(False)
    #test2(True)
    #test3(False, False)
    #test3(False, True )
    #test3(True , False)
    #test3(True , True )

if __name__ == '__main__' and len(sys.argv) >= 2:
    test3(False, True, sys.argv[1], sys.argv[2])
