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
                 mixture=LOLS.MIX_PER_ROLL, temperature=0.1, save_log = False, writer=None):
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
        self.squared_loss = 0.
        self.dev_t = []
        self.dev_a = []
        self.dev_actions = []
        self.dev_imp_weight = []
        self.dev_costs = []
        self.pred_act_cost = []
        # Contains the value differences predicted at each time-step
        self.pred_vd = []
        self.prev_state = None
        self.counter = 0
        self.action_count = 0
        self.per_step_count = np.zeros(200)
        self.save_log = save_log
        if save_log == True:
            self.writer = writer

    def forward(self, state):
        self.per_step_count[self.t] += 1
        if self.t is None or self.t == []:
            self.ref_flag = self.eval_ref()
            self.T = state.horizon()
            self.init_state = self.policy.features(state).data
            if self.deviation == 'single':
                self.dev_t.append(np.random.choice(range(self.T))+1)
            self.t = 0
            self.dev_t = []
            self.pred_act_cost = []
            self.dev_costs = []
            self.dev_actions = []
            self.dev_a = []
            self.dev_imp_weight = []
            self.pred_vd = []

        self.t += 1

        if self.t > 1:
            reward = torch.Tensor([[state.reward(self.t-2),self.t]])
            transition_tuple = torch.cat([self.prev_state, self.policy.features(state).data, reward], dim=1)
            pred_vd = self.vd_regressor(transition_tuple)
            self.pred_vd.append(pred_vd)
            self.pred_act_cost.append(pred_vd.data.numpy())
        self.prev_state = self.policy.features(state).data

        a_pol = self.policy(state)
        a_costs = self.policy.predict_costs(state)
        if self.reference is not None:
            a_ref = self.reference(state)
        # else:   # Use the exploration policy as reference if ref is None
        #     a_ref, _ = self.do_exploration(a_costs, list(state.actions)[:])

        # deviate
        if self.ref_flag and self.reference is not None:
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
                # if self.save_log == True:
                #     self.writer.add_scalar('CB/action_prob/'+f'{self.t-1}', 1.0/iw, self.per_step_count[self.t-1])
                a = dev_a if isinstance(dev_a, int) else dev_a.data[0,0]
                self.dev_t.append(self.t)
                self.dev_a.append(a)
                self.dev_actions.append(list(state.actions)[:])
                self.dev_imp_weight.append(iw)
                self.dev_costs.append(a_costs)            
        else:
            assert False, 'Unknown deviation strategy'
        return a

    def get_objective(self, loss0, final_state=None, actor_grad=True):
        loss0 = float(loss0)
        self.counter += 1
        loss_fn = nn.SmoothL1Loss(size_average=False)
        total_loss_var = 0.
        reward = torch.Tensor([[final_state.reward(self.t - 1), self.t]])
        transition_tuple = torch.cat([self.prev_state, self.policy.features(final_state).data, reward], dim=1)
        pred_vd = self.vd_regressor(transition_tuple)
        self.pred_vd.append(pred_vd)
        self.pred_act_cost.append(pred_vd.data.numpy())
        pred_value = self.ref_critic(self.init_state)
        if self.save_log == True:
            self.writer.add_scalar('trajectory_loss', loss0, self.counter)
            self.writer.add_scalar('predicted_loss', pred_value, self.counter)
        if self.ref_flag or self.reference is None:
            loss = self.ref_critic.update(pred_value, torch.Tensor([[loss0]]))
            total_loss_var += loss
        if self.dev_t is not None:
            for dev_t, dev_a, dev_actions, dev_imp_weight, dev_costs in zip(self.dev_t, self.dev_a, self.dev_actions, self.dev_imp_weight, self.dev_costs):
                if dev_costs is None or dev_imp_weight == 0.:
                    continue
                dev_costs_data = dev_costs.data if isinstance(dev_costs, Var) else \
                                 dev_costs.data() if isinstance(dev_costs, macarico.policies.bootstrap.BootstrapCost) else \
                                 None

                assert dev_costs_data is not None
                self.build_truth_vector(self.pred_act_cost[dev_t-1], dev_a, dev_imp_weight, dev_costs_data)
                importance_weight = 1
                if self.learning_method in [BanditLOLS.LEARN_MTR, BanditLOLS.LEARN_MTR_ADVANTAGE]:
                    dev_actions = [dev_a if isinstance(dev_a, int) else dev_a.data[0,0]]
                    importance_weight = dev_imp_weight
                loss_var = self.policy.update(dev_costs, self.truth, dev_actions)
                loss_var *= importance_weight
                if actor_grad == True:
                    total_loss_var += loss_var

                a = dev_a if isinstance(dev_a, int) else dev_a.data[0,0]
                self.squared_loss = (loss0 - dev_costs_data[a]) ** 2
        prefix_sum = list(accumulate(self.pred_act_cost))
        if self.deviation == 'single':
            # Only update VD regressor for timesteps after the deviation
            for dev_t in range(self.dev_t[0]-1, self.t-1):
                residual_loss = loss0 - pred_value.data.numpy() - (prefix_sum[dev_t] - self.pred_act_cost[dev_t])
                total_loss_var += self.vd_regressor.update(self.pred_vd[dev_t], residual_loss)
        elif self.deviation == 'multiple' or self.ref_flag:
            # Update VD regressor using all timesteps
            for dev_t in range(self.t-1):
                residual_loss = loss0 - pred_value.data.numpy() - (prefix_sum[dev_t] - self.pred_act_cost[dev_t])
                residual_loss = np.clip(residual_loss, -1, 1)
                total_loss_var += self.vd_regressor.update(self.pred_vd[dev_t], torch.Tensor(residual_loss))
                if self.save_log == True:
                    self.writer.add_scalar('TDIFF-predicted_tdiff/'+ f'{dev_t}', self.pred_act_cost[dev_t], self.per_step_count[dev_t])
                    self.writer.add_scalar('TDIFF-residual_loss/'+f'{dev_t}', residual_loss, self.per_step_count[dev_t])
        self.use_ref.step()
        self.eval_ref.step()
        self.t, self.dev_t, self.dev_a, self.dev_actions, self.dev_imp_weight, self.dev_costs, self.pred_vd, self.pred_act_cost, self.rollout = [None] * 9
        return total_loss_var
