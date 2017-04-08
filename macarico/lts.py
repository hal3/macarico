import torch
from torch.autograd import Variable

class LTS(object):
    def __init__(self):
        self.zero_objective()

    def zero_objective(self):
        self.objective = Variable(torch.Tensor([0.]))

    def get_objective(self):
        return self.objective
        
    def train(self, task, input):
        pass

    def act(self, state, a_ref=None):
        pass

    
