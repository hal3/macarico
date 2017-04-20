"""
Annealing schedules.
"""
from __future__ import division
from scipy.special import expit as sigmoid


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
