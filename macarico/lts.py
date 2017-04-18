from __future__ import division

import torch
from torch.autograd import Variable

class LTS(object):
    def __init__(self):
        self.zero_objective()
        self.current_pass = 0

    def zero_objective(self):
        self.objective = Variable(torch.Tensor([0.]))

    def get_objective(self):
        return self.objective

    def backward(self, *args, **kw):
        self.get_objective().backward(*args, **kw)

    def train(self, task, input):
        pass

    def new_pass(self):
        self.current_pass += 1
        pass

    def act(self, state, a_ref=None):
        pass
