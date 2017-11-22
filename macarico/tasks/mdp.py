from __future__ import division, generators, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import macarico.util as util
from macarico.util import Var, Varng
import macarico

def sample_from(l):
    r = sum((p for _, p in l)) * np.random.random()
    last_chance = None
    for k, p in l:
        if last_chance is None: last_chance = k
        r -= p
        if r <= 0: return k
    return last_chance

class MDPExample(object):
    def __init__(self, initial, transitions, costs, T):
        self.states = set([s for s, _ in initial])
        self.n_actions = 0
        for _, actions in transitions.items():
            for a, subsequent in actions.items():
                self.n_actions = max(a, self.n_actions)
                for s1, p in subsequent:
                    if p > 0:
                        self.states.add(s1)
        self.n_actions += 1
        self.initial = initial
        self.transitions = transitions
        self.costs = costs
        self.T = T

    def mk_env(self):
        return MDP(self)

class MDP(macarico.Env):
    def __init__(self, example):
        macarico.Env.__init__(self, example.n_actions)
        self.ex = example
        self.T = example.T
        self.s0 = sample_from(self.ex.initial)

    def _run_episode(self, policy):
        cost = 0
        self.s = self.s0
        self.trajectory = []
        self._trajectory = []
        for self.t in range(self.ex.T):
            self.actions = self.ex.transitions[self.s].keys()
            a = policy(self)
            s1 = sample_from(self.ex.transitions[self.s][a])
            cost += self.ex.costs(self.s, a, s1)
            self.s = s1
            self.trajectory.append(a)
            #self._trajectory.append(a)
        self.cost = cost
        return self.trajectory

    def _rewind(self):
        pass

class MDPLoss(macarico.Loss):
    def __init__(self):
        super(MDPLoss, self).__init__('cost')

    def evaluate(self, ex, env):
        return env.cost
    
class DeterministicReference(macarico.Reference):
    def __init__(self, pi_ref):
        self.pi_ref = pi_ref

    def __call__(self, state):
        return self.pi_ref(state.s)

    def set_min_costs_to_go(self, state, costs):
        a_star = self.pi_ref(state)
        costs.zero_()
        costs += 1
        costs[a_star] = 0

class MDPFeatures(macarico.StaticFeatures):
    def __init__(self, n_states, noise_rate=0):
        macarico.StaticFeatures.__init__(self, n_states)
        self.n_states = n_states
        self.noise_rate = noise_rate
        self._t = nn.Linear(1,1,bias=False)

    def _forward(self, state):
        f = util.zeros(self._t.weight, 1, 1, self.n_states)
        if np.random.random() > self.noise_rate:
            f[0, 0, state.s] = 1
        return Varng(f)

    def __call__(self, state): return self.forward(state)

def make_ross_mdp(T=100, reset_prob=0):
    initial = [(0, 1/3), (1, 1/3)]
    #               s    a    s' p()
    half_rp = reset_prob/2
    default = 1-reset_prob
    transitions = { 0: { 0: [(1, default), (0, half_rp), (2, half_rp)],
                         1: [(2, default), (0, half_rp), (1, half_rp)] },
                    1: { 0: [(2, default), (0, half_rp), (1, half_rp)],
                         1: [(1, default), (0, half_rp), (2, half_rp)] },
                    2: { 0: [(1, default), (1, half_rp), (2, half_rp)],
                         1: [(2, default), (0, half_rp), (2, half_rp)] } }

    def pi_ref(s):
        if isinstance(s, MDP):
            s = s.s
        # expert: s0->a0 s1->a1 s2->a0
        if s == 0: return 0
        if s == 1: return 1
        if s == 2: return 0
        assert False
        
    def costs(s, a, s1):
        # this is just Cmax=1 whenever we disagree with expert, and c=0 otherwise
        return 0 if a == pi_ref(s) else 1
    
    return MDPExample(initial, transitions, costs, T), \
           DeterministicReference(pi_ref)
