from __future__ import division

import numpy as np
import dynet as dy
import macarico

def sample_from(l):
    r = sum((p for _, p in l)) * np.random.random()
    last_chance = None
    for k, p in l:
        if last_chance is None: last_chance = k
        r -= p
        if r <= 0: return k
    return last_chance

class FSMExample(object):
    def __init__(self, initial, transitions, costs, T):
        self.states = set([s for s, _ in initial])
        self.n_actions = 0
        for _, actions in transitions.iteritems():
            for a, subsequent in actions.iteritems():
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
        return FSM(self)

class FSM(macarico.Env):
    def __init__(self, example):
        self.ex = example
        self.T = example.T
        self.n_actions = example.n_actions
        self.s0 = sample_from(self.ex.initial)

    def run_episode(self, policy):
        cost = 0
        self.s = self.s0
        self.trajectory = []
        self.output = []
        for self.t in xrange(self.ex.T):
            self.actions = self.ex.transitions[self.s].keys()
            a = policy(self)
            s1 = sample_from(self.ex.transitions[self.s][a])
            cost += self.ex.costs(self.s, a, s1)
            self.s = s1
            self.trajectory.append(a)
            #self.output.append(a)
        self.cost = cost
        return self.trajectory

    def rewind(self):
        pass

class FSMLoss(macarico.Loss):
    def __init__(self):
        super(FSMLoss, self).__init__('cost')

    def evaluate(self, ex, env):
        return env.cost
    
class DeterministicReference(macarico.Reference):
    def __init__(self, pi_ref):
        self.pi_ref = pi_ref

    def __call__(self, state):
        return self.pi_ref(state.s)

    def set_min_costs_to_go(self, state, costs):
        a_star = self.pi_ref(state)
        costs += 1
        costs[a_star] = 0

class FSMFeatures(macarico.Features):
    def __init__(self, n_states, noise_rate=0):
        self.n_states = n_states
        self.noise_rate = noise_rate
        macarico.Features.__init__(self, 's', self.n_states)

    def forward(self, state):
        f = np.zeros((1, self.n_states))
        if np.random.random() > self.noise_rate:
            f[0, state.s] = 1
        return dy.inputTensor(f)

    def __call__(self, state): return self.forward(state)

