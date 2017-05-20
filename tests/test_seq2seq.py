from __future__ import division
import random
import torch
import testutil
testutil.reseed()

from macarico.annealing import ExponentialAnnealing, stochastic
from macarico.lts.maximum_likelihood import MaximumLikelihood
from macarico.lts.dagger import DAgger
from macarico.tasks.seq2seq import Example, Seq2Seq
from macarico.features.sequence import RNNFeatures, BOWFeatures, AverageAttention, FrontBackAttention, SoftmaxAttention
from macarico.features.actor import TransitionRNN
from macarico.policies.linear import LinearPolicy

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
    )
    
if __name__ == '__main__':
    for attention in ['FrontBackAttention', 'AverageAttention', 'SoftmaxAttention']:
        for features in ['RNNFeatures', 'BOWFeatures']:
            test1(attention, features)

