from __future__ import division, generators, print_function

import random
import macarico

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
import macarico.util
from collections import Counter
import scipy.optimize
from macarico.annealing import Averaging, NoAnnealing, stochastic
from macarico.lts.lols import BanditLOLS, LOLS

class Reslope(BanditLOLS):
    def __init__(self, reference, policy, p_ref,
                 learning_method=BanditLOLS.LEARN_DR,
                 exploration=BanditLOLS.EXPLORE_BOLTZMANN, explore=1.0,
                 mixture=LOLS.MIX_PER_ROLL, temperature=1.):
        super(Reslope, self).__init__(policy=policy, reference=reference, exploration=exploration, mixture=mixture)
        self.reference = reference
        self.policy = policy
        self.learning_method = learning_method
        self.exploration = exploration
        self.temperature = temperature
        assert self.learning_method in range(BanditLOLS._LEARN_MAX), \
            'unknown learning_method, must be one of BanditLOLS.LEARN_*'
        assert self.exploration in range(BanditLOLS._EXPLORE_MAX), \
            'unknown exploration, must be one of BanditLOLS.EXPLORE_*'

        if mixture == LOLS.MIX_PER_ROLL:
            use_ref = p_ref()
            self.use_ref = lambda: use_ref
        else:
            self.use_ref = p_ref
        if isinstance(explore, float):
            explore = stochastic(NoAnnealing(explore))
        self.explore = explore
        self.t = None
        self.dev_t = []
        self.dev_a = []
        self.dev_actions = []
        self.dev_imp_weight = []
        self.dev_costs = []
        self.squared_loss = 0.
        self.pred_act_cost = []

    def __call__(self, state):
        if self.t is None or self.t == []:
            self.t = 0
            self.dev_costs = []
            self.pred_act_cost = []

        self.t += 1

        a_ref = self.reference(state) if self.reference is not None else None
        a_pol = self.policy(state)
        a_costs = self.policy.predict_costs(state)

        # deviate
        a = None
        if not self.explore(): # exploit
            a = a_ref if self.use_ref() else a_pol
        else:
            dev_a, iw = self.do_exploration(a_costs, state.actions)
            a = dev_a if isinstance(dev_a, int) else dev_a.data[0,0]

            if self.dev_t is None:
                self.dev_t = []
            self.dev_t.append(self.t)
            if self.dev_a is None:
                self.dev_a = []
            self.dev_a.append(a)
            if self.dev_actions is None:
               self.dev_actions = []
            self.dev_actions.append(list(state.actions)[:])
            if self.dev_imp_weight is None:
                self.dev_imp_weight = []
            self.dev_imp_weight.append(iw)
            if self.dev_costs is None:
                self.dev_costs = []
            self.dev_costs.append(a_costs)

        
        a_costs_data = a_costs.data if isinstance(a_costs, Var) else \
                       a_costs.data() if isinstance(a_costs, macarico.policies.bootstrap.BootstrapCost) else \
                         None
        self.pred_act_cost.append(a_costs_data.numpy()[a])
        return a

    def update(self, loss0):
        total_loss_var = 0.
        for dev_t, dev_a, dev_actions, dev_imp_weight, dev_costs in zip(self.dev_t, self.dev_a, self.dev_actions, self.dev_imp_weight, self.dev_costs):
            if dev_costs is None or dev_imp_weight == 0.:
                continue
            dev_costs_data = dev_costs.data if isinstance(dev_costs, Var) else \
                             dev_costs.data() if isinstance(dev_costs, macarico.policies.bootstrap.BootstrapCost) else \
                             None
            assert dev_costs_data is not None

            loss = loss0 - (sum(self.pred_act_cost) - self.pred_act_cost[dev_t-1])
            truth = self.build_cost_vector(loss, dev_a, dev_imp_weight, dev_costs_data)
            importance_weight = 1
            if self.learning_method in [BanditLOLS.LEARN_MTR, BanditLOLS.LEARN_MTR_ADVANTAGE]:
                dev_actions = [dev_a if isinstance(dev_a, int) else dev_a.data[0,0]]
                importance_weight = dev_imp_weight
            loss_var = self.policy.forward_partial_complete(dev_costs, truth, dev_actions)
            loss_var *= importance_weight
            total_loss_var += loss_var

            a = dev_a if isinstance(dev_a, int) else dev_a.data[0,0]
            self.squared_loss = (loss - dev_costs_data[a]) ** 2
        if not isinstance(total_loss_var, float):
            total_loss_var.backward()
