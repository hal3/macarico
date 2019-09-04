from __future__ import division, generators, print_function

import numpy as np
from torch.autograd import Variable as Var

import macarico
import macarico.util
from macarico.annealing import NoAnnealing, stochastic
from macarico.lts.lols import BanditLOLS, LOLS


class Reslope(BanditLOLS):
    def __init__(self, reference, policy, p_ref,
                 update_method=BanditLOLS.LEARN_DR,
                 exploration=BanditLOLS.EXPLORE_BOLTZMANN,
                 deviation='multiple', explore=1.0,
                 mixture=LOLS.MIX_PER_ROLL, temperature=1.):
        super(Reslope, self).__init__(policy=policy, reference=reference, exploration=exploration, mixture=mixture,
                                      update_method=update_method, temperature=temperature)
        self.reference = reference
        self.policy = policy
        self.update_method = update_method
        self.exploration = exploration
        self.temperature = temperature
        assert self.update_method in range(BanditLOLS._LEARN_MAX), \
            'unknown update_method, must be one of BanditLOLS.LEARN_*'
        assert self.exploration in range(BanditLOLS._EXPLORE_MAX), \
            'unknown exploration, must be one of BanditLOLS.EXPLORE_*'

        # if mixture == LOLS.MIX_PER_ROLL:
        # use_ref = p_ref()
        # self.use_ref = lambda: use_ref
        # else:
        self.use_ref = p_ref
        if isinstance(explore, float):
            explore = stochastic(NoAnnealing(explore))
        self.deviation = deviation
        self.explore = explore
        self.rollout = None
        self.t = None
        self.T = None
        self.squared_loss = 0.
        self.dev_t = []
        self.dev_a = []
        self.dev_actions = []
        self.dev_imp_weight = []
        self.dev_costs = []
        self.pred_act_cost = []

    def forward(self, state):
        if self.t is None or self.t == []:
            self.T = state.horizon()
            if self.deviation == 'single':
                self.dev_t.append(np.random.choice(range(self.T))+1)
            self.t = 0
            self.dev_t = []
            self.pred_act_cost = []
            self.dev_costs = []
            self.dev_actions = []
            self.dev_a = []
            self.dev_imp_weight = []
        self.t += 1
        a_ref = self.reference(state) if self.reference is not None else None
        a_pol = self.policy(state)
        a_costs = self.policy.predict_costs(state)
        # deviate
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
            if not self.explore():  # exploit
                a = a_ref if self.use_ref() else a_pol
            else:
                dev_a, iw = self.do_exploration(a_costs, state.actions)
                a = dev_a if isinstance(dev_a, int) else dev_a.data[0, 0]
                self.dev_t.append(self.t)
                self.dev_a.append(a)
                self.dev_actions.append(list(state.actions)[:])
                self.dev_imp_weight.append(iw)
                self.dev_costs.append(a_costs)
        else:
            assert False, 'Unknown deviation strategy'
        a_costs_data = a_costs.data if isinstance(a_costs, Var) else \
            a_costs.data() if isinstance(a_costs, macarico.policies.bootstrap.BootstrapCost) else \
                None
        self.pred_act_cost.append(a_costs_data.numpy()[a])
#        print('a_costs_data: ', a_costs_data, 'a: ', a, ' a_pol: ', a_pol)
        return a

    def get_objective(self, loss0, final_state=None):
        loss0 = np.float32(loss0)
#        print('Loss: ', loss0, '\tPredicted sum: ', sum(self.pred_act_cost))
        total_loss_var = 0.
        if self.dev_t is not None:
            for dev_t, dev_a, dev_actions, dev_imp_weight, dev_costs in zip(self.dev_t, self.dev_a, self.dev_actions,
                                                                            self.dev_imp_weight, self.dev_costs):
                if dev_costs is None or dev_imp_weight == 0.:
                    continue
                dev_costs_data = dev_costs.data if isinstance(dev_costs, Var) else \
                    dev_costs.data() if isinstance(dev_costs, macarico.policies.bootstrap.BootstrapCost) else \
                        None
                assert dev_costs_data is not None

                loss = loss0 - (np.sum(self.pred_act_cost) - self.pred_act_cost[dev_t - 1])
                self.build_truth_vector(loss, dev_a, dev_imp_weight, dev_costs_data)
                importance_weight = 1
                if self.update_method in [BanditLOLS.LEARN_MTR, BanditLOLS.LEARN_MTR_ADVANTAGE]:
                    dev_actions = [dev_a if isinstance(dev_a, int) else dev_a.data[0, 0]]
                    importance_weight = dev_imp_weight
                loss_var = self.policy.update(dev_costs, self.truth, dev_actions)
                loss_var *= importance_weight
                total_loss_var += loss_var

                a = dev_a if isinstance(dev_a, int) else dev_a.data[0, 0]
                self.squared_loss = (loss - dev_costs_data[a]) ** 2
            # if not isinstance(total_loss_var, float):
            # total_loss_var.backward()
        self.use_ref.step()
        self.t, self.dev_t, self.dev_a, self.dev_actions, self.dev_imp_weight, self.dev_costs, self.pred_act_cost, self.rollout = [None] * 8
        return total_loss_var
