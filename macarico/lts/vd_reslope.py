from __future__ import division, generators, print_function

from itertools import accumulate

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable as Var

import macarico
import macarico.util
from macarico.annealing import NoAnnealing, stochastic
from macarico.lts.lols import BanditLOLS, LOLS


class VD_Reslope(BanditLOLS):
    def __init__(self, reference, policy, ref_critic, vd_regressor, 
                 p_ref, eval_ref, learning_method=BanditLOLS.LEARN_DR,
                 exploration=BanditLOLS.EXPLORE_BOLTZMANN,  
                 deviation='multiple', explore=1.0,
                 mixture=LOLS.MIX_PER_ROLL, temperature=1.):
        super(VD_Reslope, self).__init__(policy=policy, reference=reference, exploration=exploration, mixture=mixture)
        self.reference = reference
        self.policy = policy
        self.ref_critic = ref_critic
        self.vd_regressor = vd_regressor
        self.learning_method = learning_method
        self.exploration = exploration
        self.temperature = temperature
        assert self.learning_method in range(BanditLOLS._LEARN_MAX), \
            'unknown learning_method, must be one of BanditLOLS.LEARN_*'
        assert self.exploration in range(BanditLOLS._EXPLORE_MAX), \
            'unknown exploration, must be one of BanditLOLS.EXPLORE_*'

        # if mixture == LOLS.MIX_PER_ROLL:
            # use_ref = p_ref()
            # self.use_ref = lambda: use_ref
        # else:
        self.use_ref = p_ref
        self.eval_ref = eval_ref
        self.ref_flag = 0
        self.init_state = None
        if isinstance(explore, float):
            explore = stochastic(NoAnnealing(explore))
        self.deviation = deviation
        self.explore = explore
        self.rollout = None
        self.t = None
        self.T = None
        self.dev_t = []
        self.dev_a = []
        self.dev_actions = []
        self.dev_imp_weight = []
        self.dev_costs = []
        self.squared_loss = 0.
        # Contains the value differences predicted at each time-step
        self.pred_act_cost = []
        self.pred_vd = []
        self.prev_state = None

    def forward(self, state):

        if self.t is None or self.t == []:
            self.ref_flag = self.eval_ref()
            self.T = state.horizon()
            self.init_state = self.policy.features(state).data
            self.dev_t = []
            if self.deviation == 'single':
                self.dev_t.append(np.random.choice(range(self.T))+1)
            self.t = 0
            self.pred_act_cost = []
            self.dev_costs = []
            self.dev_actions = []
            self.dev_a = []
            self.dev_imp_weight = []
            self.pred_vd = []

        self.t += 1

        if self.t > 1:
            transition_tuple = torch.cat([self.prev_state, self.policy.features(state).data], dim=1)
            pred_vd = self.vd_regressor(transition_tuple)
            self.pred_vd.append(pred_vd)
            self.pred_act_cost.append(pred_vd.data.numpy())
        self.prev_state = self.policy.features(state).data

        a_pol = self.policy(state)
        a_costs = self.policy.predict_costs(state)
        if self.reference is not None:
            a_ref = self.reference(state)
        else:   # Use the exploration policy as reference if ref is None
            a_ref, _ = self.do_exploration(a_costs, list(state.actions)[:])

        # deviate
        if self.ref_flag:
            return a_ref
        if self.deviation == 'single':
            if self.t == self.dev_t[0]:
                a = a_pol
                if self.explore():
                    dev_a, dev_imp_weight = self.do_exploration(a_costs, list(state.actions)[:])
                    a = dev_a if isinstance(dev_a, int) else dev_a.data[0,0]
                    self.dev_a.append(a)
                    self.dev_imp_weight.append(dev_imp_weight)
                    self.dev_actions.append(list(state.actions)[:])
                    self.dev_costs.append(a_costs)
            elif self.t < self.dev_t[0]:
                a = a_pol
            else:
                if self.mixture == LOLS.MIX_PER_STATE or self.rollout is None:
                    self.rollout = self.use_ref()
                a = a_ref if self.rollout else a_pol
        elif self.deviation == 'multiple':
            a = None
            # exploit
            if not self.explore():
                a = a_ref if self.use_ref() else a_pol
            else:
                dev_a, iw = self.do_exploration(a_costs, state.actions)
                a = dev_a if isinstance(dev_a, int) else dev_a.data[0,0]
                self.dev_t.append(self.t)
                self.dev_a.append(a)
                self.dev_actions.append(list(state.actions)[:])
                self.dev_imp_weight.append(iw)
                self.dev_costs.append(a_costs)            
        else:
            assert False, 'Unknown deviation strategy'
        print(self.t,'\t', a, '\t', pred_vd.data.numpy())
        return a

    def get_objective(self, loss0):
        loss0 = float(loss0)
        loss_fn = nn.SmoothL1Loss(size_average=False)
        total_loss_var = 0.
        # print('Loss: ', loss0, '\tPredicted sum: ', sum(self.pred_act_cost))
        # TODO: Need to add last transition for computing the value difference
        # transition_tuple = torch.cat([self.prev_state, self.policy.features(state).data], dim=1)
        # pred_vd = self.vd_regressor.predict_costs(transition_tuple)
        self.pred_vd.append(pred_vd)
        self.pred_act_cost.append(pred_vd.data.numpy())
        pred_value = self.ref_critic(self.init_state)
        if self.ref_flag:
            total_loss_var += self.ref_critic.update(pred_value, loss0)
        if self.dev_t is not None:
            for dev_t, dev_a, dev_actions, dev_imp_weight, dev_costs in zip(self.dev_t, self.dev_a, self.dev_actions, self.dev_imp_weight, self.dev_costs):
                if dev_costs is None or dev_imp_weight == 0.:
                    continue
                dev_costs_data = dev_costs.data if isinstance(dev_costs, Var) else \
                                 dev_costs.data() if isinstance(dev_costs, macarico.policies.bootstrap.BootstrapCost) else \
                                 None

                assert dev_costs_data is not None

                print('self.pred_act_cost: ', self.pred_act_cost)
                print('len(self.pred_act_cost): ', len(self.pred_act_cost))
                print('dev_t: ', dev_t)
                truth = self.build_truth_vector(self.pred_act_cost[dev_t], dev_a, dev_imp_weight, dev_costs_data)
                importance_weight = 1
                if self.learning_method in [BanditLOLS.LEARN_MTR, BanditLOLS.LEARN_MTR_ADVANTAGE]:
                    dev_actions = [dev_a if isinstance(dev_a, int) else dev_a.data[0,0]]
                    importance_weight = dev_imp_weight
                loss_var = self.policy.update(dev_costs, self.truth, dev_actions)
                loss_var *= importance_weight
                total_loss_var += loss_var

                a = dev_a if isinstance(dev_a, int) else dev_a.data[0,0]
                self.squared_loss = (loss0 - dev_costs_data[a]) ** 2
        prefix_sum = accumulate(self.pred_act_cost)
        if self.deviation == 'single':
            # Only update VD regressor for timesteps after the deviation
            for dev_t in range(self.dev_t[0]-1, self.t-1):
                residual_loss = loss0 - pred_value.data.numpy() - (prefix_sum[dev_t] - self.pred_act_cost[dev_t])
                total_loss_var += self.vd_regressor.update(self.pred_vd[dev_t], residual_loss)
        elif self.deviation == 'multiple' or self.ref_flag:
            # Update VD regressor using all timesteps
            for dev_t in range(self.t-1):
                residual_loss = loss0 - pred_value.data.numpy() - (prefix_sum[dev_t] - self.pred_act_cost[dev_t])
                total_loss_var += self.vd_regressor.update(self.pred_vd[dev_t], residual_loss)
        self.use_ref.step()
        self.eval_ref.step()
        self.t, self.dev_t, self.dev_a, self.dev_actions, self.dev_imp_weight, self.dev_costs, self.pred_vd, self.pred_act_cost, self.rollout = [None] * 9
        return total_loss_var
