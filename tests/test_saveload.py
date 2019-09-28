import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var

import macarico.util
macarico.util.reseed()

from macarico.annealing import ExponentialAnnealing, stochastic
from macarico.lts.maximum_likelihood import MaximumLikelihood
from macarico.tasks.sequence_labeler import Example, HammingLoss, HammingLossReference
from macarico.features.sequence import RNNFeatures, AttendAt
from macarico.features.actor import TransitionRNN
from macarico.policies.linear import LinearPolicy

def run_train(n_types, n_labels, data):
    actor = TransitionRNN([RNNFeatures(n_types)], [AttendAt()], n_labels)
    policy = LinearPolicy(actor, n_labels)
    print 'training'
    _, model = macarico.util.trainloop(
        training_data   = data[:len(data)//2],
        dev_data        = data[len(data)//2:],
        policy          = policy,
        Learner         = lambda: MaximumLikelihood(HammingLossReference(), policy),
        losses          = HammingLoss(),
        optimizer       = torch.optim.Adam(policy.parameters(), lr=0.01),
        n_epochs        = 2,
        train_eval_skip = 1,
        returned_parameters = 'best',
    )
    print 'evaluating learned model: %g' % \
        macarico.util.evaluate(data, policy, HammingLoss())
    policy.load_state_dict(model)
    print 'evaluating learned model: %g' % \
        macarico.util.evaluate(data, policy, HammingLoss())
    return model

def test_restore(n_types, n_labels, data, model):
    actor = TransitionRNN([RNNFeatures(n_types)], [AttendAt()], n_labels)
    policy = LinearPolicy(actor, n_labels)
    print 'evaluating new model: %g' % \
        macarico.util.evaluate(data, policy, HammingLoss())
    policy.load_state_dict(model)
    print 'evaluating restored model: %g' % \
        macarico.util.evaluate(data, policy, HammingLoss())
    
def test_save_load():
    n_types, n_labels = 10, 4
    data = [Example(x, y, n_labels)
            for x, y in macarico.util.make_sequence_mod_data(100, 5, n_types, n_labels)]

    model = run_train(n_types, n_labels, data)
    test_restore(n_types, n_labels, data, model)

    print 'writing model to disk'
    torch.save(model, 'test_saveload.model')

    print 'reading model from disk'
    new_model = torch.load('test_saveload.model')
    test_restore(n_types, n_labels, data, new_model)
    
if __name__ == '__main__':
    test_save_load()
