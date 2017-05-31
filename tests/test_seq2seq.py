from __future__ import division
import random
import sys
from collections import Counter
import torch
import testutil
testutil.reseed()

import numpy as np

from macarico.annealing import ExponentialAnnealing, stochastic
from macarico.lts.maximum_likelihood import MaximumLikelihood
from macarico.lts.dagger import DAgger
from macarico.tasks.seq2seq import Example, Seq2Seq
from macarico.features.sequence import RNNFeatures, BOWFeatures, AverageAttention, FrontBackAttention, SoftmaxAttention
from macarico.features.actor import TransitionRNN
from macarico.policies.linear import LinearPolicy

from nlp_data import read_parallel_data, read_embeddings

def ngrams(words):
    c = Counter()
    for l in range(4):
        for ng in zip(*[words]*(l+1)):
            c[ng] += 1
    return c

class Bleu(testutil.CustomEvaluator):
    def __init__(self):
        super(Bleu, self).__init__('bleu', corpus_level=True, maximize=True)
        self.sys = np.zeros(4)
        self.cor = np.zeros(4)
        self.len_sys = 0
        self.len_ref = 0

    def reset(self):
        self.sys = np.zeros(4)
        self.cor = np.zeros(4)
        self.len_sys = 0
        self.len_ref = 0
        
    def evaluate(self, truth, prediction):
        assert truth.labels[-1] == 0  # </s>
        self.len_ref += len(truth.labels) - 1
        self.len_sys += len(prediction)

        ref = ngrams(truth.labels[:-1])
        sys = ngrams(prediction)
        for ng, count in sys.iteritems():
            l = len(ng)-1
            self.sys[l] += count
            self.cor[l] += min(count, ref[ng])

        precision = self.cor / (self.sys + 1e-6)
        brev = min(1., np.exp(1 - self.len_ref / self.len_sys)) if self.len_sys > 0 else 0
        return 100 * brev * precision.prod()

def test1(attention_type, feature_type):
    print ''
    print 'testing seq2seq with %s / %s' % (attention_type, feature_type)
    print ''
    
    n_types = 8
    data = testutil.make_sequence_reversal_data(100, 3, n_types)
    # make EOS=0 and add one to all outputs
    n_labels = 1 + n_types
    data = [Example(X, [y+1 for y in Y] + [0], n_labels) \
            for X,Y in data]

    d_hid = 50
    features = None
    attention = None
    
    if feature_type == 'BOWFeatures':
        features = BOWFeatures(n_types, output_field='tokens_feats')
    elif feature_type == 'RNNFeatures':
        features = RNNFeatures(n_types)
        
    if attention_type == 'FrontBackAttention':
        attention = FrontBackAttention()
    elif attention_type == 'SoftmaxAttention':
        attention = SoftmaxAttention(features, d_hid)
    elif attention_type == 'AverageAttention':
        attention = AverageAttention()
    
    tRNN = TransitionRNN([features], [attention], n_labels, d_hid=d_hid)
    policy = LinearPolicy( tRNN, n_labels )

    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    p_rollin_ref = stochastic(ExponentialAnnealing(0.99))
    
    print 'eval ref: %s' % testutil.evaluate(data, lambda s: s.reference()(s))
    
    testutil.trainloop(
        training_data   = data[:len(data)//2],
        dev_data        = data[len(data)//2:],
        policy          = policy,
        Learner         = lambda ref: DAgger(ref, policy, p_rollin_ref),
        optimizer       = optimizer,
        train_eval_skip = 1,
        n_epochs        = 20,
        custom_evaluators = [Bleu()],
    )

def test_mt():
    tr, de, src_vocab, tgt_vocab = read_parallel_data('data/nc9.en',
                                                      'data/nc9.fr',
                                                      n_de=5,
                                                      min_src_freq=5,
                                                      min_tgt_freq=5,
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
    #attention = FrontBackAttention()
    tRNN = TransitionRNN([features], [attention], len(tgt_vocab), d_hid=d_hid)
    policy = LinearPolicy(tRNN, len(tgt_vocab))
    params = [p for p in policy.parameters()]# if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.001)
    p_rollin_ref = stochastic(ExponentialAnnealing(0.99))
    print 'eval ref: %s' % testutil.evaluate(de, lambda s: s.reference()(s))

    testutil.trainloop(
        training_data     = tr,
        dev_data          = de,
        policy            = policy,
        Learner           = lambda ref: MaximumLikelihood(ref, policy), #DAgger(ref, policy, p_rollin_ref),
        optimizer         = optimizer,
#        train_eval_skip   = max(1, len(tr)//20),
        n_epochs          = 20,
    )
        

if __name__ == '__main__' and len(sys.argv) == 1:
    for attention in ['FrontBackAttention', 'AverageAttention', 'SoftmaxAttention']:
        for features in ['RNNFeatures', 'BOWFeatures']:
            test1(attention, features)

if __name__ == '__main__' and len(sys.argv) == 2 and sys.argv[1] == 'mt':
    test_mt()
