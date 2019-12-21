from itertools import accumulate

import numpy as np
import torch
from vowpalwabbit import pyvw

from macarico import util
from macarico.annealing import NoAnnealing, stochastic
from macarico.lts.lols import BanditLOLS, LOLS


class VwPrep(BanditLOLS):
    def __init__(self, reference, policy, actor, p_ref=stochastic(NoAnnealing(0)),
                 exploration=BanditLOLS.EXPLORE_BOLTZMANN, explore=1.0,
                 mixture=LOLS.MIX_PER_ROLL, save_log=False, attach_time=False, expb=0):
        super(VwPrep, self).__init__(policy=policy, reference=reference, exploration=exploration, mixture=mixture,
                                     expb=expb)
        self.reference = reference
        self.policy = policy
        self.vw_ref_critic = pyvw.vw(quiet=True)
        self.vw_vd_regressor = pyvw.vw(quiet=True)
        self.exploration = exploration
        assert self.exploration in range(BanditLOLS._EXPLORE_MAX), \
            'unknown exploration, must be one of BanditLOLS.EXPLORE_*'
        # TODO handle mixture with ref
        self.use_ref = p_ref
        self.attach_time = attach_time
        self.init_state = None
        if isinstance(explore, float):
            explore = stochastic(NoAnnealing(explore))
        self.explore = explore
        self.rollout = None
        self.t = None
        self.T = None
        self.dev_ex = []
        self.dev_t = []
        self.dev_a = []
        self.dev_actions = []
        self.dev_imp_weight = []
        self.dev_costs = []
        self.pred_act_cost = []
        self.transition_ex = []
        # Contains the value differences predicted at each time-step
        self.pred_vd = []
        self.prev_state = None
        self.save_log = save_log
        # TODO why 200?
        self.per_step_count = np.zeros(200)
        self.counter = 0
        if save_log:
            self.action_count = 0
            self.critic_losses = []
            self.td_losses = []
        self.actor = actor
        self.total_sq_loss = 0.0
        self.total_vd_sq_loss = 0.0

    def forward(self, state):
        self.per_step_count[self.t] += 1
        if self.t is None or self.t == []:
            self.T = state.horizon()
            self.init_state = self.actor(state).data
            self.t = 0
            self.dev_ex = []
            self.dev_t = []
            self.pred_act_cost = []
            self.dev_costs = []
            self.dev_actions = []
            self.dev_a = []
            self.dev_imp_weight = []
            self.pred_vd = []
            self.transition_ex = []

        self.t += 1

        if self.t > 1:
            if self.attach_time:
                curr_loss = torch.Tensor([[state.loss(self.t-2), self.t]])
            else:
                curr_loss = torch.Tensor([[state.loss(self.t-2)]])
            transition_tuple = torch.cat([self.prev_state, self.actor(state).data, curr_loss], dim=1)
            transition_example = util.feature_vector_to_vw_string(transition_tuple)
            self.transition_ex.append(transition_example)
            pred_vd = self.vw_vd_regressor.predict(transition_example)
            self.pred_vd.append(pred_vd)
            self.pred_act_cost.append(pred_vd)
        self.prev_state = self.actor(state).data

        a_pol, a_prob = self.policy.stochastic(state)
        ex = util.feature_vector_to_vw_string(self.actor(state))
        self.dev_ex.append(ex)
        self.dev_t.append(self.t)
        self.dev_a.append(a_pol)
        self.dev_actions.append(list(state.actions)[:])
        self.dev_imp_weight.append(a_prob)
        a_costs = self.policy.predict_costs(state)
        self.dev_costs.append(a_costs)
        return a_pol

    def get_objective(self, loss0, final_state=None):
        loss0 = float(loss0)
        self.counter += 1
        total_loss_var = 0.
        if self.attach_time:
            terminal_loss = torch.Tensor([[final_state.loss(self.t - 1), self.t]])
        else:
            terminal_loss = torch.Tensor([[final_state.loss(self.t - 1)]])
        transition_tuple = torch.cat([self.prev_state, self.actor(final_state).data, terminal_loss], dim=1)
        transition_example = util.feature_vector_to_vw_string(transition_tuple)
        self.transition_ex.append(transition_example)
        pred_vd = self.vw_vd_regressor.predict(transition_example)
        self.pred_vd.append(pred_vd)
        self.pred_act_cost.append(pred_vd)
        ex = str(loss0) + util.feature_vector_to_vw_string(self.init_state)
        pred_value = self.vw_ref_critic.predict(ex)
        prefix_sum = list(accumulate(self.pred_act_cost))
        sq_loss = (pred_value - loss0) ** 2
        self.total_sq_loss += sq_loss
        self.vw_ref_critic.learn(ex)
        if self.dev_t is not None:
            for dev_t, dev_a, dev_actions, dev_imp_weight, dev_costs, ex in zip(self.dev_t, self.dev_a,
                                                                                self.dev_actions, self.dev_imp_weight,
                                                                                self.dev_costs, self.dev_ex):
                # residual_loss = loss0 - pred_value - (prefix_sum[dev_t-1] - self.pred_act_cost[dev_t-1])
                # residual_loss = loss0 - pred_value.data.numpy() - (prefix_sum[dev_t-1] - self.pred_act_cost[dev_t-1])
                bandit_loss = self.pred_act_cost[dev_t-1]
#                print('bandit loss: ', bandit_loss)
#                bandit_loss = loss0
                self.policy.update(dev_a, bandit_loss, dev_imp_weight, ex)
                a = dev_a if isinstance(dev_a, int) else dev_a.data[0,0]
        # Update VD regressor using all timesteps
        for dev_t in range(self.t):
            residual_loss = loss0 - pred_value - (prefix_sum[dev_t] - self.pred_act_cost[dev_t])
            vd_sq_loss = (residual_loss - pred_vd)**2
            self.total_vd_sq_loss += vd_sq_loss
#            print('squared loss: ', vd_sq_loss, ' avg: ', self.total_vd_sq_loss/float(self.counter))
            transition_example = str(residual_loss) + self.transition_ex[dev_t]
            self.vw_vd_regressor.learn(transition_example)
        self.use_ref.step()
        self.t, self.dev_t, self.dev_a, self.dev_actions, self.dev_imp_weight, self.dev_costs, self.pred_vd, self.pred_act_cost, self.rollout, self.dev_ex, self.transition_ex = [None] * 11
        return total_loss_var
