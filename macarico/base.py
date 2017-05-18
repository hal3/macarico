import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Env(object):
    def __init__(self, n_actions):
        self.n_actions = n_actions
        
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

class Reference(Policy):
    def __call__(self, state):
        raise NotImplementedError('abstract')
    
    def set_min_costs_to_go(self, state, cost_vector):
        # optional, but required by some learning algorithms (eg aggrevate)
        raise NotImplementedError('abstract')
