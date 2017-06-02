from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Env(object):
    r"""An implementation of an environment; aka a search task or MDP.

    Args:
        n_actions: the number of unique actions available to a policy
                   in this Env (actions are numbered [0, n_actions))

    Must provide a `run_episode(policy)` function that performs a
    complete run through this environment, acting according to
    `policy`.

    May optionally provide a `rewind` function that some learning
    algorithms (e.g., LOLS) requires.
    """

    def __init__(self, n_actions):
        self.n_actions = n_actions

    def run_episode(self, policy):
        pass
    
    def rewind(self):
        raise NotImplementedError('abstract')
    
class Policy(object):
    r"""A `Policy` is any function that contains a `__call__` function that
    maps states to actions."""
    def __call__(self, state):
        raise NotImplementedError('abstract')

class Features(object):
    r"""`Features` are any function that map a state (an instance of `Env`)
    to a pytorch `Variable` tensor. The dimension of the feature
    representation tensor should be (1, `dim`), where `Features.dim`
    stores the dimensionality. `Features` must also name themselves,
    in order for policies to know "where to look." They do this by
    providing a `field` name. If

    The `forward` function computes the features. You must either
    write `forward` or `_forward`. If you provide the latter, the
    module will automatically memoize the feature computation. If you
    don't use cached features (i.e., `field=None`) then you must
    provide `forward` yourself."""
    def __init__(self, field, dim):
        self.field = field
        self.dim = dim
        
    def _forward(self, state):
        raise NotImplementedError('abstract method not defined.')

    def forward(self, state):
        if self.field is None:
            raise NotImplementedError('if `Features.field` is None, you must implement your own `forward` function.')

        # check to see if computation is cached; if not, compute it
        if not hasattr(state, self.field) or \
           getattr(state, self.field) is None:
            # run the computation
            res = self._forward(state)
            setattr(state, self.field, res)

        # return the cached version
        return getattr(state, self.field)

class Learner(object):
    r"""A `Learner` behaves identically to a `Policy`, but does "stuff"
    internally to, eg., compute gradients through pytorch's `backward`
    procedure. Not all learning algorithms can be implemented this way
    (e.g., LOLS) but most can (DAgger, reinforce, etc.)."""
    
    def __call__(self, state):
        raise NotImplementedError('abstract method not defined.')

class Loss(object):
    def __init__(self, name, corpus_level=False):
        self.name = name
        self.corpus_level = corpus_level
        self.count = 0
        self.total = 0

    def evaluate(self, truth, prediction):
        raise NotImplementedError('abstract')

    def reset(self):
        self.count = 0
        self.total = 0

    def __call__(self, truth, prediction):
        val = self.evaluate(truth, prediction)
        if self.corpus_level:
            self.total = val
            self.count = 1
        else:
            self.total += val
            self.count += 1
        return self.get()

    def get(self):
        return self.total / self.count
    
class Reference(Policy):
    r"""A `Reference` is a special type of `Policy` that may use the ground
    truth to provide supervision. In many algorithms the `Reference`
    is considered to be the oracle policy (e.g., DAgger), but for some
    it is enough that it is a "good" policy (e.g., LOLS). Some
    algorithms do not use a `Reference` (e.g., reinforce).

    All `Reference`s must provide a `__call__` function that maps
    states (represented as an `Env`) to actions (just like `Policy`s).

    Some Leaners also assume that the `Reference` can provide a
    function `set_min_costs_to_go` for efficiency purposes.
    `set_min_costs_to_go` takes a `state` and a `cost_vector` (of size
    `n_actions`), and must fill in the cost-to-go for all actions if
    this reference were followed until the end of time."""
    def __call__(self, state):
        raise NotImplementedError('abstract')
    
    def set_min_costs_to_go(self, state, cost_vector):
        # optional, but required by some learning algorithms (eg aggrevate)
        raise NotImplementedError('abstract')

class Attention(object):

    r""" It is usually the case that the `Features` one wants to compute
    are a function of only some part of the input at any given time
    step. FOr instance, in a sequence labeling task, one might only
    want to look at the `Features` of the word currently being
    labeled. Or in a machine translation task, one might want to have
    dynamic, differentiable softmax-style attention.

    For static `Attention`, the class must define its `arity`: the
    number of places that it looks (e.g., one in sequence labeling).
    """
    
    arity = 0   # int=number of foci; None=attention (vector of length |input|)

    def __init__(self, field):
        self.field = field

    def __call__(self, state):
        raise NotImplementedError('abstract')
