# define various annealing rates
class Annealing:
    pass

class NoAnnealing(Annealing):
    def __init__(self, value):
        self.value = value
        
    def __call__(self, T):
        return self.value
        
class ExponentialAnnealing(Annealing):
    def __init__(self, alpha, lower_bound=0., upper_bound=1.):
        self.alpha = alpha
        self.lower_bound = lower_bound
        self.width = upper_bound - lower_bound
        self.last_T = None
        
    def __call__(self, T):
        self.last_T = T
        return self.lower_bound + self.width * (self.alpha ** T)

class LinearAnnealing(Annealing):
    def __init__(self, slope, lower_bound=0., upper_bound=1.):
        self.slope = slope
        self.lower_bound = lower_bound
        self.width = upper_bound - lower_bound
        
    def __call__(self, T):
        return self.lower_bound + self.width * max(0.,1. - self.slope * T)

class InverseSigmoidAnnealing(Annealing):
    def __init__(self, kappa, lower_bound=0., upper_bound=1.):
        self.kappa = kappa
        self.lower_bound = lower_bound
        self.width = upper_bound - lower_bound
        
    def __call__(self, T):
        return self.lower_bound + self.width * self.kappa / (self.kappa + np.exp(T / self.kappa))

class UserAnnealing(Annealing):
    def __init__(self, f):
        self.f = f
        
    def __call__(self, T):
        return self.f(T)
