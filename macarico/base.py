from __future__ import division, generators, print_function
import sys
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable as Var

if True:
    # check version
    vers = torch.__version__.split('.')
    major = int(vers[0])
    minor = int(vers[1])
    assert major == 0 and minor >= 4, \
        "sorry, macarico requires pytorch version >= 0.4, you have %s" % torch.__version__

def check_intentional_override(class_name, fn_name, override_bool_name, obj, *fn_args):
    if not getattr(obj, override_bool_name): # self.OVERRIDE_RUN_EPISODE:
        try:
            getattr(obj, fn_name)(*fn_args)
        except NotImplementedError:
            print("*** warning: %s %s\n"
                  "*** does not seem to have an implemented '%s'\n"
                  "*** perhaps you overrode %s by accident?\n"
                  "*** if not, suppress this warning by setting\n"
                  "*** %s=True\n"
                  % (class_name, type(obj),
                     fn_name, fn_name[1:],
                     override_bool_name),
                  file=sys.stderr)
        except:
            pass            

class Example(object):
    def __init__(self, X=None, Y=None):
        self.X = X  # generic input
        self.Y = Y  # generic output
        self.Yhat = None  # prediction

    def __str__(self):
        return '{ X: %s, Y: %s, Yhat: %s }' % (self.X, self.Y, self.Yhat)

    def __repr__(self):
        return str(self)

    def input_str(self):
        return self._simple_str(self.X)

    def output_str(self):
        return self._simple_str(self.Y)

    def prediction_str(self):
        return self._simple_str(self.Yhat)

    def _simple_str(self, A):
        if A is None: return '?'
        if isinstance(A, list): return ' '.join(map(str, A))
        return str(A)

        
    
class Env(object):
    r"""An implementation of an environment; aka a search task or MDP.

    Args:
        n_actions: the number of unique actions available to a policy
                   in this Env (actions are numbered [0, n_actions))

    Must provide a `_run_episode(policy)` function that performs a
    complete run through this environment, acting according to
    `policy`.

    May optionally provide a `_rewind` function that some learning
    algorithms (e.g., LOLS) requires.
    """
    OVERRIDE_RUN_EPISODE = False
    OVERRIDE_REWIND = False
    
    def __init__(self, n_actions, T, example=None):
        self.n_actions = n_actions
        self.T = T
        self.example = Example() if example is None else example
        self._trajectory = []
        check_intentional_override('Env', '_run_episode', 'OVERRIDE_RUN_EPISODE', self, None)
        check_intentional_override('Env', '_rewind', 'OVERRIDE_REWIND', self)
    
    def horizon(self):
        return self.T

    def timestep(self):
        return len(self._trajectory)

    def output(self):
        return self._trajectory
    
    def run_episode(self, policy):
        def _policy(state):
            assert self.timestep() < self.horizon()
            a = policy(state)
            self._trajectory.append(a)
            return a
        self.rewind(policy)
        out = self._run_episode(_policy)
        self.example.Yhat = out if out is not None else self._trajectory
        return self.example.Yhat
    
    def input_x(self):
        return self.example.X

    def rewind(self, policy):
        self._trajectory = []
        if hasattr(policy, 'new_run'):
            policy.new_run()
        self._rewind()
        
    def _run_episode(self, policy):
        raise NotImplementedError('abstract')
    
    def _rewind(self):
        raise NotImplementedError('abstract')

class TypeMemory(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.param = Parameter(torch.zeros(1))
    
class StaticFeatures(nn.Module):
    r"""`StaticFeatures` are any function that map an `Env` to a
    tensor. The dimension of the feature representation tensor should
    be (1, N, `dim`), where `N` is the length of the input, and
    `dim()` returns the dimensionality.

    The `forward` function computes the features."""

    OVERRIDE_FORWARD = False
    def __init__(self, dim):
        nn.Module.__init__(self)
        self.dim = dim
        self._current_env = None
        self._features = None
        self._batched_features = None
        self._batched_lengths = None
        self._my_id = '%s #%d' % (type(self), id(self))
        self._typememory = TypeMemory()
        check_intentional_override('StaticFeatures', '_forward', 'OVERRIDE_FORWARD', self, None)

    def _forward(self, env):
        raise NotImplementedError('abstract')

    def forward(self, env):
        # check to see if batched computation is done
        if self._batched_features is not None and \
           hasattr(env, '_stored_batch_features') and \
           self._my_id in env._stored_batch_features:
            # just get the stored features
            i = env._stored_batch_features[self._my_id]
            assert 0 <= i and i < self._batched_features.shape[0]
            assert self._batched_lengths[i] <= self._batched_features.shape[1]
            l = self._batched_lengths[i]
            self._features = self._batched_features[i,:l,:].unsqueeze(0)
            
        if self._features is None:
            self._features = self._forward(env)
            assert self._features.dim() == 3
            assert self._features.shape[0] == 1
            assert self._features.shape[2] == self.dim
        assert self._features is not None
        return self._features

    def _forward_batch(self, envs):
        raise NotImplementedError('abstract')

    def forward_batch(self, envs):
        if self._batched_features is not None:
            return self._batched_features, self._batched_lengths

        try:
            res = self._forward_batch(envs)
            assert isinstance(res, tuple)
            self._batched_features, self._batched_lengths = res
            if self._batched_features.shape[0] != len(envs):
                import ipdb; ipdb.set_trace()
        except NotImplementedError:
            pass

        if self._batched_features is None:
            # okay, they didn't implement it, so let's do it for them!
            bf = [self._forward(env) for env in envs]
            for x in bf:
                assert x.dim() == 3
                assert x.shape[0] == 1
                assert x.shape[2] == self.dim
            max_len = max((x.shape[1] for x in bf))
            self._batched_features = Var(self._typememory.param.data.new(len(envs), max_len, self.dim).zero_())
            self._batched_lengths = []
            for i, x in enumerate(bf):
                self._batched_features[i,:x.shape[1],:] = x
                self._batched_lengths.append(x.shape[1])
            
        assert self._batched_features.shape[0] == len(envs)

        # remember for each environment which id it is
        for i, env in enumerate(envs):
            if not hasattr(env, '_stored_batch_features'):
                env._stored_batch_features = dict()
            env._stored_batch_features[self._my_id] = i

            # sanity check
            #x1 = self._batched_features[i,0:env.N,:]
            #x2 = self._forward(env)[0,0:env.N,:]
            #print(x1.shape, x2.shape, env.N, type(self))
            #if x1.shape != x2.shape:
            #    import ipdb; ipdb.set_trace()
            #if (x1-x2).abs().sum().data[0] > 1e-4:
            #    import ipdb; ipdb.set_trace()

        return self._batched_features, self._batched_lengths
    
    
class Actor(nn.Module):
    r"""An `Actor` is a module that computes features dynamically as a policy runs."""
    OVERRIDE_FORWARD = False
    def __init__(self, n_actions, dim, attention):
        nn.Module.__init__(self)
        self._current_env = None
        self._features = None
        self.n_actions = n_actions

        self.dim = dim
        self.attention = nn.ModuleList(attention)
        self._T = None
        self._last_t = 0

        for att in attention:
            if att.actor_dependent:
                att.set_actor(self)

        self._typememory = TypeMemory()

        check_intentional_override('Actor', '_forward', 'OVERRIDE_FORWARD', self, None)
                
    def reset(self):
        self._last_t = 0
        self._T = None
        self._features = None
        self._reset()

    def _reset(self):
        pass
        
    def _forward(self, state, x):
        raise NotImplementedError('abstract')
        
    def hidden(self):
        raise NotImplementedError('abstract')
        
    def forward(self, env):
        if self._features is None or self._T is None:
            self._T = env.horizon()
            self._features = [None] * self._T
            self._last_t = 0
            
        t = env.timestep()
        # we want to make sure that we "keep up" with the
        # environment. so we'll store self._last_t, and if t >
        # self._last_t+1 then bad news
        assert t <= self._last_t+1
        assert t >= self._last_t
        self._last_t = t
        
        assert self._features is not None

        assert t >= 0, 'expect t>=0, bug?'
        assert t < self._T, ('%d=t < T=%d' % (t, self._T))
        assert t < len(self._features)
        
        if self._features[t] is not None:
            return self._features[t]
        
        assert t == 0 or self._features[t-1] is not None

        x = []
        for att in self.attention:
            x += att(env)

        ft = self._forward(env, x)
        assert ft.dim() == 2
        assert ft.shape[0] == 1
        assert ft.shape[1] == self.dim
        
        self._features[t] = ft
        return self._features[t]

class Policy(nn.Module):
    r"""A `Policy` is any function that contains a `forward` function that
    maps states to actions."""
    def __init__(self):
        nn.Module.__init__(self)
    
    def forward(self, state):
        raise NotImplementedError('abstract')

    """
    cases where we need to reset:
    - 0. new minibatch. this means reset EVERYTHING.
    - 1. new example in a minibatch. this means reset dynamic and static _features, but not _batched_features
    - 2. replaying the current example. this means reset dynamic ONLY.
    flipped around:
    - Actors are reset in all cases
    - _features is reset in 0 and 1
    - _batched_features is reset in 0
    """

    def new_minibatch(self): self._reset_some(0, True)
    def new_example(self): self._reset_some(1, True)
    def new_run(self): self._reset_some(2, True)
    
    def _reset_some(self, reset_type, recurse):
        for module in self.modules():
            if isinstance(module, Actor): # always reset dynamic features
                module.reset()
            if isinstance(module, StaticFeatures):
                if reset_type == 0 or reset_type == 1:
                    module._features = None
                if reset_type == 0:
                    #if module._batched_features is not None:
                        #print('unset _batched_features ', module)
                        #if isinstance(module, Policy):
                        #    import ipdb; ipdb.set_trace()
                    module._batched_features = None
            #elif module != self and isinstance(module, Policy) and recurse:
            #    module._reset_some(reset_type, False)

class StochasticPolicy(Policy):
    def stochastic(self, state):
        # returns a:int, p(a):Var(float)
        raise NotImplementedError('abstract')

    def sample(self, state):
        return self.stochastic(state)[0]

class CostSensitivePolicy(Policy):
    OVERRIDE_UPDATE = False
    
    def __init__(self):
        nn.Module.__init__(self)
        check_intentional_override('CostSensitivePolicy', '_update', 'OVERRIDE_UPDATE', self, None, None)
        
    def predict_costs(self, state):
        # raturns Var([float]*n_actions)
        raise NotImplementedError('abstract')

    def costs_to_action(self, state, pred_costs):
        # this is roughly a duplicate of util.argmin
        if isinstance(pred_costs, Var): pred_costs = pred_costs.data
        if state.actions is None or len(state.actions) == 0 or len(state.actions) == pred_costs.shape[0]:
            return pred_costs.min(0)[1][0]
        i = None
        for a in state.actions:
            if i is None or pred_costs[a] < pred_costs[i]:
                i = a
        return i

    
    def update(self, state_or_pred_costs, truth, actions=None):
        if isinstance(state_or_pred_costs, Env):
            assert actions is None
            actions = state_or_pred_costs.actions
            state_or_pred_costs = self.predict_costs(state_or_pred_costs)
        return self._update(state_or_pred_costs, truth, actions)

    def _update(self, pred_costs, truth, actions=None):
        raise NotImplementedError('abstract')
        
                
class Learner(Policy):
    r"""A `Learner` behaves identically to a `Policy`, but does "stuff"
    internally to, eg., compute gradients through pytorch's `backward`
    procedure. Not all learning algorithms can be implemented this way
    (e.g., LOLS) but most can (DAgger, reinforce, etc.)."""
    def forward(self, state):
        raise NotImplementedError('abstract method not defined.')

    def get_objective(self, loss):
        raise NotImplementedError('abstract method not defined.')

class LearningAlg(nn.Module):
    def __call__(self, env):
        raise NotImplementedError('abstract method not defined.')
    
class Loss(object):
    OVERRIDE_EVALUATE = False
    def __init__(self, name, corpus_level=False):
        self.name = name
        self.corpus_level = corpus_level
        self.count = 0
        self.total = 0
        check_intentional_override('Loss', 'evaluate', 'OVERRIDE_EVALUATE', self, None, None)

    def evaluate(self, example):
        raise NotImplementedError('abstract')

    def reset(self):
        self.count = 0.0
        self.total = 0.0

    def __call__(self, example):
        assert example.Yhat is not None
        val = self.evaluate(example)
        if self.corpus_level:
            self.total = val
            self.count = 1
        elif val is not None:
            self.total += val
            self.count += 1.0
        return self.get()

    def get(self):
        return self.total / self.count if self.count > 0 else 0
    
class Reference(object):
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

class Attention(nn.Module):
    r""" It is usually the case that the `Features` one wants to compute
    are a function of only some part of the input at any given time
    step. For instance, in a sequence labeling task, one might only
    want to look at the `Features` of the word currently being
    labeled. Or in a machine translation task, one might want to have
    dynamic, differentiable softmax-style attention.

    For static `Attention`, the class must define its `arity`: the
    number of places that it looks (e.g., one in sequence labeling).
    """
    
    OVERRIDE_FORWARD = False
    
    arity = 0   # int=number of attention targets; None=attention (vector of length |input|)
    actor_dependent = False

    def __init__(self, features):
        nn.Module.__init__(self)
        self.features = features
        self.dim = (self.arity or 1) * self.features.dim
        check_intentional_override('Attention', '_forward', 'OVERRIDE_FORWARD', self, None)

    def forward(self, state):
        fts = self._forward(state)
        dim_sum = 0
        if self.arity is None: assert len(fts) == 1
        if self.arity is not None: assert len(fts) == self.arity
        for ft in fts:
            assert ft.dim() == 2
            assert ft.shape[0] == 1
            dim_sum += ft.shape[1]
        assert dim_sum == self.dim
        return fts
        
    def _forward(self, state):
        raise NotImplementedError('abstract')

    def set_actor(self, actor):
        raise NotImplementedError('abstract')
    
    def make_out_of_bounds(self):
        oob = Parameter(torch.Tensor(1, self.features.dim))
        oob.data.zero_()
        return oob

class Torch(nn.Module):
    def __init__(self, features, dim, layers):
        nn.Module.__init__(self)
        self.features = features
        self.dim = dim
        self.torch_layers = layers if isinstance(layers, nn.ModuleList) else \
                            nn.ModuleList(layers) if isinstance(layers, list) else \
                            nn.ModuleList([layers])

    def forward(self, x):
        x = self.features(x)
        for l in self.torch_layers:
            x = l(x)
        return x

