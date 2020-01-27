from itertools import accumulate

import numpy as np
import torch
from vowpalwabbit import pyvw

from macarico import util
from macarico.lts.lols import BanditLOLS, LOLS


class VwPrep(BanditLOLS):
    def __init__(self, policy, actor, vd_regressor, ref_critic, learner_type='prep'):
        super(VwPrep, self).__init__(policy=policy)
        self.policy = policy
        self.vw_ref_critic = ref_critic
        self.vw_vd_regressor = vd_regressor
        self.learner_type = learner_type
        self.init_state = None
        self.t = None
        self.T = None
        self.dev_ex = []
        self.dev_fts = []
        self.dev_t = []
        self.dev_a = []
        self.dev_prob = []
        self.pred_act_cost = []
        self.transition_ex = []
        # Contains the value differences predicted at each time-step
        self.prev_state = None
        # TODO why 200?
        self.counter = 0
        self.actor = actor
        self.total_sq_loss = 0.0
        self.total_vd_sq_loss = 0.0

    def forward(self, state):
        if self.t is None or self.t == []:
            self.T = state.horizon()
            self.init_state = self.actor(state).data
            self.t = 0
            self.dev_ex = []
            self.dev_fts = []
            self.dev_t = []
            self.pred_act_cost = []
            self.dev_a = []
            self.dev_prob = []
            self.transition_ex = []

        self.t += 1
        pred_vd = 0
        if self.t > 1:
            curr_loss = torch.Tensor([[state.loss(self.t-2)]])
            if self.learner_type == 'prep':
                transition_tuple = torch.cat([self.prev_state, self.actor(state).data, curr_loss], dim=1)
                transition_example = util.feature_vector_to_vw_string(transition_tuple)
                pred_vd = self.vw_vd_regressor.predict(transition_example)
            else:
                transition_example = util.feature_vector_to_vw_string(self.actor(state).data)
                pred_vd = self.vw_vd_regressor.predict(transition_example)
            self.transition_ex.append(transition_example)
        self.prev_state = self.actor(state).data

        a_pol, a_prob = self.policy.stochastic(state)
        if self.learner_type == 'prep' or self.learner_type == 'bootstrap':
            self.pred_act_cost.append(pred_vd)
        else:
            self.pred_act_cost.append(self.policy.predict_costs(state)[a_pol])
        ex = util.feature_vector_to_vw_string(self.actor(state))
        self.dev_fts.append(self.actor(state))
        self.dev_ex.append(ex)
        self.dev_t.append(self.t)
        self.dev_a.append(a_pol)
        self.dev_prob.append(a_prob)
        return a_pol

    def get_objective(self, loss0, final_state=None):
        # TODO cleanup and generlize beyond grid-world
        loss0 = float(loss0)
        prefix_sum = []
        sum_val = 0.0
        self.counter += 1
        terminal_loss = torch.Tensor([[final_state.loss(self.t - 1)]])
        if self.learner_type == 'prep':
            transition_tuple = torch.cat([self.prev_state, self.actor(final_state).data, terminal_loss], dim=1)
            transition_example = util.feature_vector_to_vw_string(transition_tuple)
            pred_vd = self.vw_vd_regressor.predict(transition_example)
            self.pred_act_cost.append(pred_vd)
        else:
            transition_example = util.feature_vector_to_vw_string(self.actor(final_state).data)
            pred_vd = self.vw_vd_regressor.predict(transition_example)
            self.pred_act_cost.append(pred_vd)
        self.transition_ex.append(transition_example)
        initial_state_ex = str(loss0) + util.feature_vector_to_vw_string(self.init_state)
        initial_state_value = self.vw_ref_critic.predict(initial_state_ex)
        prefix_sum = list(accumulate(self.pred_act_cost))
        sum_val = np.sum(self.pred_act_cost)
        assert self.dev_t is not None
        for dev_t, dev_a, dev_prob, dev_ex, transition_ex, dev_fts in zip(
                self.dev_t, self.dev_a, self.dev_prob, self.dev_ex, self.transition_ex, self.dev_fts):
            if self.learner_type == 'prep':
                residual_loss = loss0 - initial_state_value - (prefix_sum[dev_t-1] - self.pred_act_cost[dev_t-1])
                transition_example = str(residual_loss) + transition_ex
                self.vw_vd_regressor.learn(transition_example)
                bandit_loss = residual_loss
            elif self.learner_type == 'reslope':
                bandit_loss = loss0 - sum_val + self.pred_act_cost[dev_t-1]
            elif self.learner_type == 'mc':
                bandit_loss = final_state.loss_to_go(dev_t-1)
            elif self.learner_type == 'bootstrap':
                reg_target = final_state.loss(dev_t-1) + self.pred_act_cost[dev_t-1]
                transition_example = str(reg_target) + transition_ex
                self.vw_vd_regressor.learn(transition_example)
                bandit_loss = reg_target
            self.policy.update(dev_a, bandit_loss, dev_prob, dev_fts)
        self.vw_ref_critic.learn(initial_state_ex)
        self.t, self.dev_t, self.dev_a, self.dev_prob, self.pred_act_cost, self.dev_ex, self.transition_ex = [None] * 7
        return loss0
