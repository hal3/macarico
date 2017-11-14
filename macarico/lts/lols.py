from __future__ import division, generators, print_function

import random # TODO make this np
import numpy as np
import macarico
import macarico.util

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
from macarico.annealing import Averaging, NoAnnealing, stochastic
import macarico.policies.costeval


class LOLS(macarico.LearningAlg):
    MIX_PER_STATE, MIX_PER_ROLL = 0, 1

    def __init__(self,
                 policy,
                 reference,
                 loss_fn,
                 p_rollin_ref=NoAnnealing(0),
                 p_rollout_ref=NoAnnealing(0.5),
                 mixture=MIX_PER_ROLL,
                ):
        macarico.LearningAlg.__init__(self)
        self.policy = policy
        self.reference = reference
        self.loss_fn = loss_fn
        self.rollin_ref = stochastic(p_rollin_ref)
        self.rollout_ref = stochastic(p_rollout_ref)
        self.mixture = mixture
        self.rollout = None

    def __call__(self, example):
        self.example = example
        self.env = example.mk_env()
        n_actions = self.env.n_actions

        # compute training loss
        loss0, _, _, _ = self.run(lambda _: EpisodeRunner.LEARN)
        
        # generate backbone using rollin policy
        _, traj0, limit0, costs0 = self.run(lambda _: self.rollin_ref())
        T = len(traj0)

        # run all one step deviations
        objective = 0.
        follow_traj0 = lambda t: (EpisodeRunner.ACT, traj0[t])
        for t, pred_costs in enumerate(costs0):
            true_costs = torch.zeros(n_actions)
            rollout = TiedRandomness(self.make_rollout())
            for a in limit0[t]:
                l, _, _, _ = self.run(one_step_deviation(T, follow_traj0, rollout, t, a))
                true_costs[a] = float(l)
            true_costs -= true_costs.min()
            objective += self.policy.forward_partial_complete(pred_costs, true_costs, limit0[t])

        # run backprop
        objective_value = objective.data[0]
        if not isinstance(objective, float):
            objective.backward()

        self.rollin_ref.step()
        self.rollout_ref.step()

        return float(loss0), objective_value
            
    def run(self, run_strategy):
        self.env.rewind()
        runner = EpisodeRunner(self.policy, run_strategy, self.reference)
        self.env.run_episode(runner)
        cost = self.loss_fn.evaluate(self.example, self.env)
        return cost, runner.trajectory, runner.limited_actions, runner.costs

    def make_rollout(self):
        mk = lambda _: (EpisodeRunner.REF if self.rollout_ref() else EpisodeRunner.LEARN)
        if self.mixture == LOLS.MIX_PER_ROLL:
            rollout = mk(0)
            return lambda _: rollout
        else: # MIX_PER_STATE
            return mk
        

class BanditLOLS(macarico.Learner):
    LEARN_BIASED, LEARN_IPS, LEARN_DR, LEARN_MTR, LEARN_MTR_ADVANTAGE, _LEARN_MAX = 0, 1, 2, 3, 4, 5
    EXPLORE_UNIFORM, EXPLORE_BOLTZMANN, EXPLORE_BOLTZMANN_BIASED, EXPLORE_BOOTSTRAP, _EXPLORE_MAX = 0, 1, 2, 3, 4

    def __init__(self,
                 policy,
                 reference=None,
                 p_rollin_ref=NoAnnealing(0),
                 p_rollout_ref=NoAnnealing(0.5),
                 update_method=LEARN_MTR,
                 exploration=EXPLORE_BOLTZMANN,
                 p_explore=NoAnnealing(1.0),
                 mixture=LOLS.MIX_PER_ROLL,
                ):
        macarico.Learner.__init__(self)
        if reference is None: reference = lambda s: np.random.choice(list(s.actions))
        self.policy = policy
        self.reference = reference
        self.rollin_ref = stochastic(p_rollin_ref)
        self.rollout_ref = stochastic(p_rollout_ref)
        self.update_method = update_method
        self.exploration = exploration
        self.explore = stochastic(p_explore)
        self.mixture = mixture

        assert self.update_method in range(BanditLOLS._LEARN_MAX), \
            'unknown update_method, must be one of BanditLOLS.LEARN_*'
        assert self.exploration in range(BanditLOLS._EXPLORE_MAX), \
            'unknown exploration, must be one of BanditLOLS.EXPLORE_*'

        self.dev_t = None
        self.dev_a = None
        self.dev_actions = None
        self.dev_imp_weight = None
        self.dev_costs = None
        self.rollout = None
        self.t = None

    def forward(self, state):
        if self.t is None:
            self.T = state.horizon()
            self.dev_t = np.random.choice(range(self.T))
            self.t = 0

        a_ref = self.reference(state)
        a_pol = self.policy(state)
        if self.t == self.dev_t:
            a = a_pol
            if self.explore():
                self.dev_costs = self.policy.predict_costs(state)
                self.dev_actions = list(state.actions)[:]
                self.dev_a, self.dev_imp_weight = self.do_exploration(self.dev_costs, self.dev_actions)
                a = self.dev_a if isinstance(self.dev_a, int) else self.dev_a.data[0,0]

        elif self.t < self.dev_t:
            a = a_ref if self.rollin_ref() else a_pol
            
        else: # state.t > self.dev_t:
            if self.mixture == LOLS.MIX_PER_STATE or self.rollout is None:
                self.rollout = self.rollout_ref()
            a = a_ref if self.rollout else a_pol

        self.t += 1
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
                costs += disallow
            probs = F.softmax(- costs, dim=0)
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
            a, p = util.sample_from_np_probs(probs)
            return a, 1 / p
        assert False, 'unknown exploration strategy'


    def update(self, loss):
        loss = float(loss)
        obj = 0.
        
        if self.dev_a is not None:
            dev_costs_data = self.dev_costs.data if isinstance(self.dev_costs, Var) else \
                             self.dev_costs.data() if isinstance(self.dev_costs, macarico.policies.bootstrap.BootstrapCost) else \
                             None

            truth = self.build_cost_vector(loss, self.dev_a, self.dev_imp_weight, dev_costs_data)
            importance_weight = 1.0
            if self.update_method in [BanditLOLS.LEARN_MTR, BanditLOLS.LEARN_MTR_ADVANTAGE]:
                self.dev_actions = [self.dev_a if isinstance(self.dev_a, int) else self.dev_a.data[0,0]]
                importance_weight = self.dev_imp_weight
            loss_var = self.policy.forward_partial_complete(self.dev_costs, truth, self.dev_actions)
            loss_var *= importance_weight
            obj = loss_var.data[0]
            loss_var.backward()

        self.explore.step()
        self.rollin_ref.step()
        self.rollout_ref.step()

        self.t, self.dev_t, self.dev_a, self.dev_actions, self.dev_imp_weight, self.dev_costs, self.rollout = [None] * 7
        return obj

    def build_cost_vector(self, loss, a, imp_weight, dev_costs_data):
        costs = torch.zeros(self.policy.n_actions)
        if not isinstance(a, int): a = a.data[0,0]
        if self.update_method == BanditLOLS.LEARN_BIASED:
            costs[a] = loss
        elif self.update_method == BanditLOLS.LEARN_IPS:
            costs[a] = loss * imp_weight
        elif self.update_method == BanditLOLS.LEARN_DR:
            costs += dev_costs_data # now costs = \hat c
            costs[a] = dev_costs_data[a] + imp_weight * (loss - dev_costs_data[a])
        elif self.update_method == BanditLOLS.LEARN_MTR:
            costs[a] = loss
        elif self.update_method == BanditLOLS.LEARN_MTR_ADVANTAGE:
            costs[a] = loss - dev_costs_data.min()
        else:
            assert False, self.update_method
        return costs


class EpisodeRunner(macarico.Learner):
    REF, LEARN, ACT = 0, 1, 2

    def __init__(self, policy, run_strategy, reference=None):
        macarico.Learner.__init__(self)
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
    def __init__(self, rand):
        self.tied = {}
        self.rand = rand

    def reset(self):
        self.tied = {}

    def __call__(self, t):
        if t not in self.tied:
            self.tied[t] = self.rand(t)
        return self.tied[t]
