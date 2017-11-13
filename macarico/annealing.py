from __future__ import division, generators, print_function
"""
Annealing schedules.
"""
from scipy.special import expit as sigmoid
from random import random

class Annealing:
    "Base case."
    def __call__(self, T):
        raise NotImplementedError('abstract method not implemented.')


class NoAnnealing(Annealing):
    "Constant rate."

    def __init__(self, value):
        self.value = value

    def __call__(self, T):
        return self.value


class ExponentialAnnealing(Annealing):
    "Exponential decay within upper and lower bounds."

    def __init__(self, alpha, lower_bound=0., upper_bound=1.):
        self.alpha = alpha
        self.lower_bound = lower_bound
        self.width = upper_bound - lower_bound

    def __call__(self, T):
        return self.lower_bound + self.width * (self.alpha ** T)


class LinearAnnealing(Annealing):
    "Linear decay within upper and lower bounds."

    def __init__(self, slope, lower_bound=0., upper_bound=1.):
        self.slope = slope
        self.lower_bound = lower_bound
        self.width = upper_bound - lower_bound

    def __call__(self, T):
        return self.lower_bound + self.width * max(0.,1. - self.slope * T)


class NegativeSigmoidAnnealing(Annealing):
    "Sigmoid(-x/kappa) annealing within upper and lower bounds."

    def __init__(self, kappa, lower_bound=0., upper_bound=1.):
        self.kappa = kappa
        self.lower_bound = lower_bound
        self.width = upper_bound - lower_bound

    def __call__(self, T):
        # timv: This doesn't look correct: why so many kappas?
        #return self.lower_bound + self.width * self.kappa / (self.kappa + expit(T / self.kappa))
        return self.lower_bound + self.width * sigmoid(-T / self.kappa)


# timv: UserAnnealing: is only useful to wrap `f` in something which is a
# subtype `Annealing`. it's better to just subclass Annealing.
#class UserAnnealing(Annealing):
#    def __init__(self, f):
#        self.f = f
#    def __call__(self, T):
#        return self.f(T)


class EWMA(object):
    "Exponentially weighted moving average."

    def __init__(self, rate, initial_value = 0.0):
        self.rate = rate
        self.value = initial_value

    def update(self, x):
        if self.value is None:
            self.value = x
        else:
            self.value += self.rate*(x - self.value)

    def __call__(self):
        return self.value


class Averaging(object):
    "Simple averaging."

    def __init__(self):
        self.count = 0.0
        self.value = 0.0

    def update(self, x):
        self.count += 1.0
        self.value += float(x)

    def __call__(self):
        if self.count == 0: return 0
        return self.value / self.count

    

class stochastic(object):
    def __init__(self, inst):
        assert isinstance(inst, Annealing)
        self.inst = inst
        self.time = 1
    def step(self):
        self.time += 1
    def __call__(self):
        return random() <= self.inst(self.time)
