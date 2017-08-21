from __future__ import division
import random
import torch

import testutil
testutil.reseed()

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from macarico.annealing import ExponentialAnnealing, stochastic
from macarico.tasks.sequence_labeler import Example, HammingLoss, HammingLossReference

def test_wsj():
    print
    print '# test on wsj subset'
    import nlp_data
    tr,de,te,vocab,label_id = \
      nlp_data.read_wsj_pos('data/wsj.pos', n_tr=50, n_de=50, n_te=0)

    n_types = len(vocab)
    n_labels = len(label_id)

    print 'n_train: %s, n_dev: %s, n_test: %s' % (len(tr), len(de), len(te))
    print 'n_types: %s, n_labels: %s' % (n_types, n_labels)

    d_emb = 50
    d_rnn = 50
    d_hid = 50
    d_actemb = 5
    n_layers = 1
    bidirectional = True
    
    embed_w = nn.Embedding(n_types, d_emb)
    #from arsenal.debug import ip; ip()
    gru = nn.GRU(d_emb, d_rnn, n_layers, bidirectional)
    embed_a = nn.Embedding(n_labels, d_actemb)
    combine = nn.Linear(d_actemb + d_rnn + d_hid, d_hid)
    
    initial_h_tensor = torch.Tensor(1, d_hid)
    initial_h_tensor.zero_()
    initial_h = Parameter(initial_h_tensor)
    
    initial_ae_tensor = torch.Tensor(1, d_actemb)
    initial_ae_tensor.zero_()
    initial_ae = Parameter(initial_ae_tensor)

    predictor = nn.Linear(d_hid, n_labels)

    loss_fn = torch.nn.MSELoss(size_average=False)

    minibatch_size = 1
    
    optimizer = torch.optim.Adam(
        list(embed_w.parameters()) +
        list(gru.parameters()) +
        list(embed_a.parameters()) +
        list(combine.parameters()) +
        list(predictor.parameters()) +
        [initial_h, initial_ae]
        , lr=0.01)
    p_rollin_ref = stochastic(ExponentialAnnealing(0.99))

    n_epochs = 10

    def my_learner(ex):
        loss = 0
        N = len(ex.tokens)
        e = embed_w(Variable(torch.LongTensor(ex.tokens), requires_grad=False)).view(N, 1, -1)
        [res, _] = gru(e)
        prev_h = initial_h
        ae = initial_ae
        output = []
        for t in xrange(N):
            inputs = [ae, prev_h, e[t]]
            h = F.relu(combine(torch.cat(inputs, 1)))

            pred_vec = predictor(h)
            pred = pred_vec.data.numpy().argmin()
            output.append(pred)
            truth = torch.ones(n_labels)
            truth[ex.labels[t]] = 0
            loss += loss_fn(pred_vec, Variable(truth, requires_grad=False))

            prev_h = h
            ae = embed_a(Variable(torch.LongTensor([pred]), requires_grad=False))
        loss.backward()
        return 0, loss.data.numpy()[0]

    
    """
    for epoch in xrange(n_epochs):
        total_loss = 0
        for batch in testutil.minibatch(tr, minibatch_size, True):
            optimizer.zero_grad()
            loss = 0
            for ex in batch:
                N = len(ex.tokens)
                e = embed_w(Variable(torch.LongTensor(ex.tokens), requires_grad=False)).view(N, 1, -1)
                [res, _] = gru(e)
                prev_h = initial_h
                ae = initial_ae
                output = []
                for t in xrange(N):
                    inputs = [ae, prev_h, e[t]]
                    h = F.relu(combine(torch.cat(inputs, 1)))

                    pred_vec = predictor(h)
                    pred = pred_vec.data.numpy().argmin()
                    output.append(pred)
                    truth = torch.ones(n_labels)
                    truth[ex.labels[t]] = 0
                    loss += loss_fn(pred_vec, Variable(truth, requires_grad=False))

                    prev_h = h
                    ae = embed_a(Variable(torch.LongTensor([pred]), requires_grad=False))

            loss.backward()
            total_loss += loss.data.numpy()[0]
            optimizer.step()
        print total_loss
    """

    testutil.trainloop(
        training_data   = tr,
        dev_data        = None, #de,
#        Learner         = lambda: MaximumLikelihood(HammingLossReference(), policy),
        losses          = HammingLoss(),
        optimizer       = optimizer,
        n_epochs        = n_epochs,
        train_eval_skip = None,
        learning_alg = my_learner,
        returned_parameters='none',
    )

    
if __name__ == '__main__':
    test_wsj()
