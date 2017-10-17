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
from macarico.tasks.stringedit import Example, StringEdit, StringEditLoss, StringEditReference
from macarico.features.sequence import RNNFeatures, BOWFeatures, AverageAttention, FrontBackAttention, SoftmaxAttention, AttendAt
from macarico.features.actor import TransitionRNN
from macarico.policies.linear import LinearPolicy

from macarico.data.nlp_data import read_parallel_data, read_embeddings, Bleu

class ExactMatch(macarico.Loss):
    def __init__(self):
        super(ExactMatch, self).__init__('exact', corpus_level=False)

    def evaluate(self, truth, state):
        return 0 if truth.labels[:-1] == state.output else 1

class EvalOnDifferent(macarico.Loss):
    def __init__(self, loss):
        self.loss = loss()
        super(EvalOnDifferent, self).__init__('D_' + self.loss.name, corpus_level=self.loss.corpus_level)

    def reset(self):
        self.loss.reset()

    def evaluate(self, truth, state):
        if truth.raw_tokens == truth.raw_labels:
            return None
        return self.loss.evaluate(truth, state)

def test1():
    print ''
    print 'testing stringedit on simple data'
    print ''
    
    n_labels = 8+1
    def mk_example():
        X = list(np.random.randint(1,n_labels,10))
        Y = [1] + [x for x in X if x != 1] + [0]
        return Example(X, Y, n_labels)

    data = [mk_example() for _ in xrange(500)]

    d_hid = 50
    features = None

    dy_model = dy.ParameterCollection()
    
    features = RNNFeatures(dy_model, n_labels)
    attention = [AttendAt(), SoftmaxAttention(dy_model, features, d_hid)]
        
    #if attention_type == 'FrontBackAttention':
    #    attention = FrontBackAttention()
    #elif attention_type == 'SoftmaxAttention':
    #    attention = SoftmaxAttention(dy_model, features, d_hid)
    #elif attention_type == 'AverageAttention':
    #    attention = AverageAttention()
    #elif attention_type == '
    
    tRNN = TransitionRNN(dy_model, [features], attention, n_labels+2, d_hid=d_hid)
    policy = LinearPolicy(dy_model, tRNN, n_labels+2)

    optimizer = dy.AdamTrainer(dy_model, alpha=0.001)
    p_rollin_ref = stochastic(ExponentialAnnealing(1.0))
    
    print 'eval ref: %s' % macarico.util.evaluate(data, StringEditReference(n_labels), StringEditLoss())

    macarico.util.trainloop(
        training_data   = data[:len(data)//2],
        dev_data        = data[len(data)//2:],
        policy          = policy,
        Learner         = lambda: DAgger(StringEditReference(n_labels), policy, p_rollin_ref),
        losses          = [Bleu(), StringEditLoss()],
        optimizer       = optimizer,
        train_eval_skip = 1,
        n_epochs        = 20,
        run_per_batch   = [p_rollin_ref.step]
#        custom_evaluators = [Bleu()],
    )

def test_ofe():
    vocab = { c: n+3 for n, c in enumerate('_abcdefghijklmnopqrstuvwxyz0123456789') }
    vocab['</s>'] = 0
    vocab['<s>'] = 1
    vocab['<OOV>'] = 2
    tr, de, _, _ = read_parallel_data('data/ofe_train.src',
                                      'data/ofe_train.tgt',
                                      n_tr=2**16,
                                      n_de=200,
                                      min_src_freq=3,
                                      min_tgt_freq=3,
                                      max_src_len=20,
                                      max_tgt_len=20,
                                      max_ratio=1.5,
                                      remove_tgt_oov=False,
                                      example_class=Example,
                                      src_vocab=vocab, tgt_vocab=vocab,
                                      )
    print >>sys.stderr, 'read %d/%d train/dev sentences, |vocab|=%d' % \
        (len(tr), len(de), len(vocab))

    dy_model = dy.ParameterCollection()

    d_hid = 50
    n_types = max(vocab.values())+1
    features = RNNFeatures(dy_model, n_types, d_rnn=d_hid)
    
    attention = [AttendAt(), SoftmaxAttention(dy_model, features, d_hid)]
    tRNN = TransitionRNN(dy_model, [features], attention, n_types+2, d_hid=d_hid)
    policy = LinearPolicy(dy_model, tRNN, n_types+2)
    optimizer = dy.AdamTrainer(dy_model, alpha=0.005)
    p_rollin_ref = stochastic(ExponentialAnnealing(0.99))
    print 'eval ref: %s' % macarico.util.evaluate(de, StringEditReference(n_types), StringEditLoss())

    macarico.util.trainloop(
        training_data     = tr,
        dev_data          = de,
        policy            = policy,
        Learner           = lambda: MaximumLikelihood(StringEditReference(n_types), policy), #DAgger(ref, policy, p_rollin_ref),
        optimizer         = optimizer,
        losses            = [Bleu(), StringEditLoss(), EvalOnDifferent(Bleu), ExactMatch(), EvalOnDifferent(ExactMatch)],
        n_epochs          = 2**4,
        train_eval_skip   = len(tr)//100,
    )
        

if __name__ == '__main__' and len(sys.argv) == 1:
    test1()

if __name__ == '__main__' and len(sys.argv) == 2 and sys.argv[1] == 'ofe':
    test_ofe()
