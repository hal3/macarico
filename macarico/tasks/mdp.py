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
        self._T = T

class MDP(macarico.Env):
    def __init__(self, example):
        macarico.Env.__init__(self, example.n_actions, example._T, example)
        self.s0 = sample_from(self.example.initial)
        self.example.cost = 0

    def _run_episode(self, policy):
        cost = 0
        self.s = self.s0
        for _ in range(self.horizon()):
            self.actions = self.example.transitions[self.s].keys()
            a = policy(self)
            s1 = sample_from(self.example.transitions[self.s][a])
            cost += self.example.costs(self.s, a, s1)
            self.s = s1
        self.example.cost = cost
        return self._trajectory

    def _rewind(self):
        pass

class MDPLoss(macarico.Loss):
    def __init__(self):
        super(MDPLoss, self).__init__('cost')

    def evaluate(self, example):
        return example.cost
    
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

class MDPFeatures(macarico.DynamicFeatures):
    def __init__(self, n_states, noise_rate=0):
        macarico.DynamicFeatures.__init__(self, n_states)
        self.n_states = n_states
        self.noise_rate = noise_rate
        self._t = nn.Linear(1,1,bias=False)

    def _forward(self, state):
        f = util.zeros(self._t.weight, 1, 1, self.n_states)
        if np.random.random() > self.noise_rate:
            f[0, 0, state.s] = 1
        return Varng(f)

    def __call__(self, state): return self.forward(state)

