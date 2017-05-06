import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Env(object):
    def run_episode(self, policy):
        pass
    def rewind(self):
        raise NotImplementedError('abstract')
    
class Policy(object):
    def __call__(self, state):
        raise NotImplementedError('abstract')

class Features(object):
    def __init__(self, dim):
        self.dim = dim
    def forward(self, state):
        raise NotImplementedError('abstract method not defined.')

class LearningAlg(object):
    def __call__(self, state):
        raise NotImplementedError('abstract method not defined.')
