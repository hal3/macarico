from __future__ import division, generators, print_function
import random
import sys
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
import macarico.util as util
import macarico.data.synthetic as synth

util.reseed()

from macarico.annealing import ExponentialAnnealing, stochastic
from macarico.lts.behavioral_cloning import BehavioralCloning
from macarico.lts.dagger import DAgger
from macarico.tasks.seq2seq import Example, Seq2Seq, EditDistance, EditDistanceReference, NgramFollower
from macarico.features.sequence import RNN, BOWFeatures, EmbeddingFeatures, DilatedCNN, AverageAttention, FrontBackAttention, SoftmaxAttention, AttendAt
from macarico.actors.rnn import RNNActor
from macarico.policies.linear import CSOAAPolicy

from macarico.data.nlp_data import read_parallel_data, read_embeddings, Bleu

def test1(attention_type, feature_type):
    print('')
    print('testing seq2seq with %s / %s' % (attention_type, feature_type))
    print('')
    
    n_types = 8
    n_labels = 1 + n_types
    data = [Example([i+1 for i in x], [i+1 for i in y], n_labels) \
            for (x,y) in synth.make_sequence_reversal_data(100, 3, n_types)]

    d_hid = 50
    features = BOWFeatures(n_labels) if feature_type == 'BOWFeatures' else \
               RNN(EmbeddingFeatures(n_labels)) if feature_type == 'RNNFeatures' else \
               DilatedCNN(EmbeddingFeatures(n_labels)) if feature_type == 'DilatedCNN'  else \
               None
    attention = FrontBackAttention if attention_type == 'FrontBackAttention' else \
                SoftmaxAttention   if attention_type == 'SoftmaxAttention'   else \
                AverageAttention   if attention_type == 'AverageAttention'   else \
                None

    ref = NgramFollower # or EditDistanceReference
    actor = RNNActor([attention(features)], n_labels)
    policy = CSOAAPolicy(actor, n_labels)
    learner = DAgger(policy, ref(), p_rollin_ref=ExponentialAnnealing(0.9))
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.005)
    
    print('eval ref: %s' % util.evaluate(data, ref(), EditDistance()))
    
    util.trainloop(
        training_data   = data[:len(data)//2],
        dev_data        = data[len(data)//2:],
        policy          = policy,
        learner         = learner,
        losses          = [Bleu, EditDistance],
        optimizer       = optimizer,
        n_epochs        = 2000,
    )


if __name__ == '__main__' and len(sys.argv) == 1:
    for attention in ['AverageAttention']: #, 'FrontBackAttention', 'SoftmaxAttention']:
        for features in ['BOWFeatures', 'RNNFeatures']:
            test1(attention, features)
