from __future__ import division
import random
import sys
from collections import Counter
import dynet as dy
import macarico.util
macarico.util.reseed()

import numpy as np

from macarico.annealing import ExponentialAnnealing, stochastic
from macarico.lts.maximum_likelihood import MaximumLikelihood
from macarico.lts.dagger import DAgger
from macarico.tasks.seq2seq import Example, Seq2Seq, EditDistance, EditDistanceReference
from macarico.features.sequence import RNNFeatures, BOWFeatures, AverageAttention, FrontBackAttention, SoftmaxAttention
from macarico.features.actor import TransitionRNN
from macarico.policies.linear import LinearPolicy

from macarico.data.nlp_data import read_parallel_data, read_embeddings, Bleu

def test1(attention_type, feature_type):
    print ''
    print 'testing seq2seq with %s / %s' % (attention_type, feature_type)
    print ''
    
    n_types = 8
    data = macarico.util.make_sequence_reversal_data(100, 3, n_types)
    # make EOS=0 and add one to all outputs
    n_labels = 1 + n_types
    data = [Example(X, [y+1 for y in Y] + [0], n_labels) \
            for X,Y in data]

    d_hid = 50
    features = None
    attention = None

    dy_model = dy.ParameterCollection()
    
    if feature_type == 'BOWFeatures':
        features = BOWFeatures(dy_model, n_types, output_field='tokens_feats')
    elif feature_type == 'RNNFeatures':
        features = RNNFeatures(dy_model, n_types)
        
    if attention_type == 'FrontBackAttention':
        attention = FrontBackAttention()
    elif attention_type == 'SoftmaxAttention':
        attention = SoftmaxAttention(dy_model, features, d_hid)
    elif attention_type == 'AverageAttention':
        attention = AverageAttention()
    
    tRNN = TransitionRNN(dy_model, [features], [attention], n_labels, d_hid=d_hid)
    policy = LinearPolicy(dy_model, tRNN, n_labels)

    optimizer = dy.AdamTrainer(dy_model, alpha=0.001)
    p_rollin_ref = stochastic(ExponentialAnnealing(0.99))
    
    print 'eval ref: %s' % macarico.util.evaluate(data, EditDistanceReference(), EditDistance())
    
    macarico.util.trainloop(
        training_data   = data[:len(data)//2],
        dev_data        = data[len(data)//2:],
        policy          = policy,
        Learner         = lambda: DAgger(EditDistanceReference(), policy, p_rollin_ref),
        losses          = [Bleu(), EditDistance()],
        optimizer       = optimizer,
        train_eval_skip = 1,
        n_epochs        = 20,
#        custom_evaluators = [Bleu()],
    )

def test_mt():
    tr, de, src_vocab, tgt_vocab = read_parallel_data('data/nc9.en',
                                                      'data/nc9.fr',
                                                      n_de=2000,
                                                      min_src_freq=1,
                                                      min_tgt_freq=1,
                                                      max_src_len=10,
                                                      max_tgt_len=10,
                                                      max_ratio=1.5,
                                                      remove_tgt_oov=True,
                                                     )
    print >>sys.stderr, 'read %d/%d train/dev sentences, |src_vocab|=%d, |tgt_vocab|=%d' % \
        (len(tr), len(de), len(src_vocab), len(tgt_vocab))

    #embeddings = read_embeddings('data/glove.6B.50d.txt.gz', src_vocab)
    #print >>sys.stderr, 'read %d embeddings' % len(embeddings)

#    tr = tr[:20]
    
    d_hid = 50
    features = RNNFeatures(len(src_vocab),
#                           initial_embeddings=embeddings,
#                           learn_embeddings=False,
                          )
    
    attention = SoftmaxAttention(features, d_hid)
    tRNN = TransitionRNN([features], [attention], len(tgt_vocab), d_hid=d_hid)
    policy = LinearPolicy(tRNN, len(tgt_vocab))
    params = [p for p in policy.parameters()]# if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.001)
    p_rollin_ref = stochastic(ExponentialAnnealing(0.99))
    print 'eval ref: %s' % macarico.util.evaluate(de, lambda s: s.reference()(s))

    macarico.util.trainloop(
        training_data     = tr[:1000],
        dev_data          = None, #de,
        policy            = policy,
        Learner           = lambda ref: MaximumLikelihood(ref, policy), #DAgger(ref, policy, p_rollin_ref),
        optimizer         = optimizer,
#        train_eval_skip   = max(1, len(tr)//20),
        n_epochs          = 1,
#        custom_evaluators = [Bleu()],
    )
        

if __name__ == '__main__' and len(sys.argv) == 1:
    for attention in ['FrontBackAttention', 'AverageAttention', 'SoftmaxAttention']:
        for features in ['RNNFeatures', 'BOWFeatures']:
            test1(attention, features)

if __name__ == '__main__' and len(sys.argv) == 2 and sys.argv[1] == 'mt':
    test_mt()
