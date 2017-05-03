from __future__ import division
import random
import torch

from macarico.lts.maximum_likelihood import MaximumLikelihood
from macarico.tasks.sequence_labeler import BiLSTMFeatures
from macarico.tasks.seq2seq import Seq2Seq, Seq2SeqFoci
from macarico import LinearPolicy

import testutil


def test1():
    n_types = 10
    data = testutil.make_sequence_reversal_data(1000, 5, n_types)
    # make EOS=0 and add one to all outputs
    n_labels = 1 + n_types
    data = [(X,[y+1 for y in Y] + [0]) for X,Y in data]

    policy = LinearPolicy(BiLSTMFeatures(Seq2SeqFoci(), n_types, n_labels), n_labels)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    Env = lambda x: Seq2Seq(x, n_labels)
    
    print 'eval ref: %s' % testutil.evaluate(Env, data, None)
    
    testutil.trainloop(
        Env             = Env,
        training_data   = data[:len(data)//2],
        dev_data        = data[len(data)//2:],
        policy          = policy,
        Learner         = lambda ref: MaximumLikelihood(ref, policy),
        optimizer       = optimizer,
        train_eval_skip = 1,
        n_epochs        = 500,
    )


if __name__ == '__main__':
    test1()
