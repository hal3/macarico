from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable

class Reference:
    def __init__(self, truth=None):
        pass

    def reset(self):
        raise Exception('reset not defined')

    def step(self, p):
        raise Exception('step not defined')

    def loss(self):
        raise Exception('loss not defined')

    def next(self):
        raise Exception('next not defined')

    def final_loss(self):
        raise Exception('final_loss no defined')


class HammingReference(Reference):
    def __init__(self, truth=None):
        self.truth = truth
        self.reset()

    def reset(self):
        self.prediction = []

    def step(self, a):
        self.prediction.append(a)

    def loss(self):
        loss = 0.
        for n,y in enumerate(self.truth):
            if n >= len(self.prediction) or y != self.prediction[n]:
                loss += 1.
        loss += max(0., len(self.prediction) - len(self.truth))
        return loss

    def next(self):
        n = len(self.prediction)
        if n >= len(self.truth):
            return 0
        return self.truth[n]

    def final_loss(self):
        return self.loss()
