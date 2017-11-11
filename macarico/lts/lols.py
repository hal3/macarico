from __future__ import division

import random # TODO make this np
import numpy as np
#import torch
#from torch.autograd import Variable
import macarico
#import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
import macarico.util
from collections import Counter
import scipy.optimize
from macarico.annealing import Averaging, NoAnnealing, stochastic

class BanditLOLS(macarico.Learner):
    MIX_PER_STATE, MIX_PER_ROLL = 0, 1
    LEARN_BIASED, LEARN_IPS, LEARN_DR, LEARN_MTR, LEARN_MTR_ADVANTAGE, _LEARN_MAX = 0, 1, 2, 3, 4, 5
    EXPLORE_UNIFORM, EXPLORE_BOLTZMANN, EXPLORE_BOLTZMANN_BIASED, EXPLORE_BOOTSTRAP, _EXPLORE_MAX = 0, 1, 2, 3, 4

    def __init__(self, reference, policy, p_rollin_ref, p_rollout_ref,
                 learning_method=LEARN_IPS,
                 exploration=EXPLORE_UNIFORM, explore=0.0,
                 mixture=MIX_PER_ROLL, temperature=1.):
        self.reference = reference
        self.policy = policy
        self.learning_method = learning_method
        self.exploration = exploration
        self.temperature = temperature
        assert self.learning_method in range(BanditLOLS._LEARN_MAX), \
            'unknown learning_method, must be one of BanditLOLS.LEARN_*'
        assert self.exploration in range(BanditLOLS._EXPLORE_MAX), \
            'unknown exploration, must be one of BanditLOLS.EXPLORE_*'

        if mixture == BanditLOLS.MIX_PER_ROLL:
            use_in_ref  = p_rollin_ref()
            use_out_ref = p_rollout_ref()
            self.rollin_ref  = lambda: use_in_ref
            self.rollout_ref = lambda: use_out_ref
        else:
            self.rollin_ref  = p_rollin_ref
            self.rollout_ref = p_rollout_ref
        if isinstance(explore, float):
            explore = stochastic(NoAnnealing(explore))
        self.explore = explore
        self.t = None
        self.dev_t = None
        self.dev_a = None
        self.dev_actions = None
        self.dev_imp_weight = None
        self.dev_costs = None
        self.squared_loss = 0.
        self.deviated = False

        super(BanditLOLS, self).__init__() # 1560 s

    def __call__(self, state):
        global global_times, certainty_tracker, num_offsets
        if self.t is None:
            self.t = 0
            self.dev_t = np.random.choice(range(state.T)) + 1

        self.t += 1
        
        a_ref = self.reference(state) if self.reference is not None else None
        a_pol = self.policy(state)
        if self.t == self.dev_t:
            a = None
            if not self.explore():
                a = a_pol
            else:
                self.dev_costs = self.policy.predict_costs(state)
                self.dev_actions = list(state.actions)[:]
                self.dev_a, self.dev_imp_weight = self.do_exploration(self.dev_costs, self.dev_actions)
                a = self.dev_a if isinstance(self.dev_a, int) else self.dev_a.data[0,0]

        else:
            a = a_ref
            if a is None or not (self.rollin_ref() if self.t < self.dev_t else self.rollout_ref()):
                a = a_pol

        return a

    def do_exploration(self, costs, dev_actions):
        # returns action and importance weight
        if self.exploration == BanditLOLS.EXPLORE_UNIFORM:
            return random.choice(list(dev_actions)), len(dev_actions)
        if self.exploration in [BanditLOLS.EXPLORE_BOLTZMANN, BanditLOLS.EXPLORE_BOLTZMANN_BIASED]:
            if len(dev_actions) != self.policy.n_actions:
                disallow = torch.zeros(self.policy.n_actions)
                for i in xrange(self.policy.n_actions):
                    if i not in dev_actions:
                        disallow[i] = 1e10
                costs += dy.inputTensor(disallow)
            probs = dy.softmax(- costs / self.temperature)
            a, p = macarico.util.sample_from_probs(probs)
            p = p.data[0]
            if self.exploration == BanditLOLS.EXPLORE_BOLTZMANN_BIASED:
                p = max(p, 1e-4)
            return a, 1 / p
        if self.exploration == BanditLOLS.EXPLORE_BOOTSTRAP:
            assert isinstance(self.policy,
                              macarico.policies.bootstrap.BootstrapPolicy) or \
                    isinstance(self.policy,
                               macarico.policies.costeval.CostEvalPolicy)
            probs = costs.get_probs(dev_actions)
            a, p = macarico.util.sample_from_np_probs(probs)
            return a, 1 / p
        assert False, 'unknown exploration strategy'


    def update(self, loss):
        #self.pred_cost_without_dev = self.pred_cost_total - self.pred_cost_dev
        self.explore.step()

        if self.dev_a is not None:
            truth = self.build_cost_vector(baseline, loss, self.dev_a, self.dev_imp_weight, self.dev_costs)
            importance_weight = 1
            old_dev_actions = self.dev_actions[:]
            if self.learning_method in [BanditLOLS.LEARN_MTR, BanditLOLS.LEARN_MTR_ADVANTAGE]:
                self.dev_actions = [self.dev_a if isinstance(self.dev_a, int) else self.dev_a.data[0,0]]
                importance_weight = self.dev_imp_weight
            #print 'diff = %s, a = %s' % (self.dev_costs.data - truth, self.dev_actions)
            #print (self.dev_costs.data - truth)[0,self.dev_actions[0]]
            loss_var = self.policy.forward_partial_complete(self.dev_costs, truth, self.dev_actions)
            self.dev_actions = old_dev_actions # TODO remove?
            loss_var *= importance_weight
            loss_var.backward()

            a = self.dev_a if isinstance(self.dev_a, int) else self.dev_a.data[0,0]
            self.squared_loss = (loss - self.dev_costs.data[a]) ** 2

    def build_cost_vector(self, baseline, loss, dev_a, imp_weight, dev_costs):
        costs = torch.zeros(self.policy.n_actions)
        a = dev_a
        if not isinstance(a, int):
            a = a.data[0,0]
        if self.learning_method == BanditLOLS.LEARN_BIASED:
            costs -= baseline
            costs[a] = loss - baseline
        elif self.learning_method == BanditLOLS.LEARN_IPS:
            costs -= baseline
            costs[a] = (loss - baseline) * imp_weight
        elif self.learning_method == BanditLOLS.LEARN_DR:
            costs += dev_costs.data # now costs = \hat c
            costs[a] = dev_costs.data[a] + imp_weight * (loss - dev_costs.data[a])
        elif self.learning_method == BanditLOLS.LEARN_MTR:
            costs[a] = loss - baseline
        elif self.learning_method == BanditLOLS.LEARN_MTR_ADVANTAGE:
            costs[a] = loss - dev_costs.data.min()
        else:
            assert False, self.learning_method
        return costs


class EpisodeRunner(macarico.Learner):
    REF, LEARN, ACT = 0, 1, 2

    def __init__(self, policy, run_strategy, reference=None):
        self.policy = policy
        self.run_strategy = run_strategy
        self.reference = reference
        self.t = 0
        self.total_loss = 0.
        self.trajectory = []
        self.limited_actions = []
        self.costs = []
        self.ref_costs = []

    def __call__(self, state):
        a_type = self.run_strategy(self.t)
        pol = self.policy(state)
        ref_costs_t = torch.zeros(self.policy.n_actions)
        self.reference.set_min_costs_to_go(state, ref_costs_t)
        self.ref_costs.append(ref_costs_t)
        if a_type == EpisodeRunner.REF:
            a = self.reference(state)
        elif a_type == EpisodeRunner.LEARN:
            a = pol
        elif isinstance(a_type, tuple) and a_type[0] == EpisodeRunner.ACT:
            a = a_type[1]
        else:
            raise ValueError('run_strategy yielded an invalid choice %s' % a_type)

        assert a in state.actions, \
            'EpisodeRunner strategy insisting on an illegal action :('

        self.limited_actions.append(list(state.actions))
        self.trajectory.append(a)
        cost = self.policy.predict_costs(state)
        self.costs.append(cost)
        self.t += 1

        return a

def one_step_deviation(T, rollin, rollout, dev_t, dev_a):
    def run(t):
        if   t == dev_t: return (EpisodeRunner.ACT, dev_a)
        elif t <  dev_t: return rollin(t)
        else:            return rollout(t)
    return run

class TiedRandomness(object):
    def __init__(self, rng=random.random):
        self.tied = {}
        self.rng = rng

    def reset(self):
        self.tied = {}

    def __call__(self, t):
        if t not in self.tied:
            self.tied[t] = self.rng()
        return self.tied[t]

def lols(ex, loss, ref, policy, p_rollin_ref, p_rollout_ref,
         mixture=BanditLOLS.MIX_PER_ROLL):
    # construct the environment
    env = ex.mk_env()
    # set up a helper function to run a single trajectory
    def run(run_strategy):
        env.rewind()
        runner = EpisodeRunner(policy, run_strategy, ref)
        env.run_episode(runner)
        cost = loss()(ex, env)
        return cost, runner.trajectory, runner.limited_actions, runner.costs

    n_actions = env.n_actions

    # construct rollin and rollout policies
    if mixture == BanditLOLS.MIX_PER_STATE:
        # initialize tied randomness for both rollin and rollout
        # TODO THIS IS BORKEN!
        rng = TiedRandomness()
        rollin_f  = lambda t: EpisodeRunner.REF if rng(t) <= p_rollin_ref  else EpisodeRunner.LEARN
        rollout_f = lambda t: EpisodeRunner.REF if rng(t) <= p_rollout_ref else EpisodeRunner.LEARN
    else:
        rollin  = EpisodeRunner.REF if p_rollin_ref()  else EpisodeRunner.LEARN
        rollout = EpisodeRunner.REF if p_rollout_ref() else EpisodeRunner.LEARN
        rollin_f  = lambda t: rollin
        rollout_f = lambda t: rollout

    # build a back-bone using rollin policy
    loss0, traj0, limit0, costs0 = run(rollin_f)
    T = env.T

    # start one-step deviations
    objective = 0. # Variable(torch.zeros(1))
    traj_rollin = lambda t: (EpisodeRunner.ACT, traj0[t])
    for t, costs_t in enumerate(costs0):
        costs = torch.zeros(n_actions)
        # collect costs for all possible actions
        for a in limit0[t]:
            l, traj, _, _ = run(one_step_deviation(T, traj_rollin, rollout_f, t, a))
            costs[a] = l
        # accumulate update
        costs -= costs.min()
        objective += policy.forward_partial_complete(costs_t, costs, limit0[t])

    # run backprop
    v = objective.data
    objective.backward()

    return v, v
