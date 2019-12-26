from itertools import accumulate

import numpy as np
import torch
from vowpalwabbit import pyvw

from macarico import util
from macarico.annealing import NoAnnealing, stochastic
from macarico.lts.lols import BanditLOLS, LOLS


class VwPrep(BanditLOLS):
    def __init__(self, reference, policy, actor, exploration=BanditLOLS.EXPLORE_BOLTZMANN, mixture=LOLS.MIX_PER_ROLL,
                 expb=0):
        super(VwPrep, self).__init__(policy=policy, reference=reference, exploration=exploration, mixture=mixture,
                                     expb=expb)
        self.policy = policy
        self.vw_ref_critic = pyvw.vw(quiet=True)
        self.vw_vd_regressor = pyvw.vw(quiet=True)
        self.exploration = exploration
        assert self.exploration in range(BanditLOLS._EXPLORE_MAX), \
            'unknown exploration, must be one of BanditLOLS.EXPLORE_*'
        self.init_state = None
        self.t = None
        self.T = None
        self.dev_ex = []
        self.dev_t = []
        self.dev_a = []
        self.dev_imp_weight = []
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
            self.dev_t = []
            self.pred_act_cost = []
            self.dev_a = []
            self.dev_imp_weight = []
            self.transition_ex = []

        self.t += 1

        if self.t > 1:
            curr_loss = torch.Tensor([[state.loss(self.t-2)]])
            transition_tuple = torch.cat([self.prev_state, self.actor(state).data, curr_loss], dim=1)
            transition_example = util.feature_vector_to_vw_string(transition_tuple)
            self.transition_ex.append(transition_example)
            pred_vd = self.vw_vd_regressor.predict(transition_example)
            self.pred_act_cost.append(pred_vd)
        self.prev_state = self.actor(state).data

        a_pol, a_prob = self.policy.stochastic(state)
        ex = util.feature_vector_to_vw_string(self.actor(state))
        self.dev_ex.append(ex)
        self.dev_t.append(self.t)
        self.dev_a.append(a_pol)
        self.dev_imp_weight.append(a_prob)
        return a_pol

    def get_objective(self, loss0, final_state=None):
        # TODO cleanup and generlize beyond grid-world
        states = np.eye(16)
        Pi = np.zeros((16, 4))
        for i, state in enumerate(states):
            Pi[i] = self.policy.distribution(state)
        # Definition of rewards for gridworld
        # Transition model
#        print('Pi: ', Pi)
#        model = final_state.model(Pi)
#        costs = final_state.costs(Pi)
        costs_function = final_state.costs_function()
        P = final_state.transition()
#        V_Pi = np.dot(np.linalg.inv(np.eye(16) - final_state.example.gamma * model), costs)
#        V_Pi = final_state.policy_eval(Pi, P, costs_function, final_state.example.gamma, theta=0.0)
#        Q_Pi = costs_function + final_state.example.gamma * P.dot(V_Pi)
#         V = []
        # Q = []
        # for max_steps in range(final_state.example.max_steps+1):
        #     V_ = final_state.policy_eval(Pi, P, costs_function, max_steps, discount_factor=final_state.example.gamma, theta=0.0)
        #     Q_ = costs_function + final_state.example.gamma * P.dot(V_)
        #     V.append(V_)
        #     Q.append(Q_)
        V, Q = final_state.fin_horizon_VI(Pi, P, costs_function, self.T, discount_factor=final_state.example.gamma)
#        print(V)
#        print(Q)
#        print('V_Pi[initial state]: ', V_Pi[3])
#        print('loss0: ', loss0)
        # For the current policy Pi, what is the distribution over different actions?
#        print(V_Pi)
        loss0 = float(loss0)
        self.counter += 1
        terminal_loss = torch.Tensor([[final_state.loss(self.t - 1)]])
        transition_tuple = torch.cat([self.prev_state, self.actor(final_state).data, terminal_loss], dim=1)
        transition_example = util.feature_vector_to_vw_string(transition_tuple)
        self.transition_ex.append(transition_example)
        pred_vd = self.vw_vd_regressor.predict(transition_example)
        self.pred_act_cost.append(pred_vd)
        initial_state_ex = str(loss0) + util.feature_vector_to_vw_string(self.init_state)
        initial_state_value = self.vw_ref_critic.predict(initial_state_ex)
        prefix_sum = list(accumulate(self.pred_act_cost))
        sq_loss = (initial_state_value - loss0) ** 2
        self.total_sq_loss += sq_loss
        assert self.dev_t is not None
        td_residual_array = []
        for dev_t, dev_a, transition_ex in zip(self.dev_t, self.dev_a, self.transition_ex):
            start_state = [float(x.split(':')[1]) for x in transition_ex.replace('|', '').strip().split()[:-1]][:16].index(1.0)
            end_state = [float(x.split(':')[1]) for x in transition_ex.replace('|', '').strip().split()[:-1]][16:].index(1.0)
            td_residual = costs_function[start_state, dev_a] + final_state.example.gamma * V[-dev_t-1][end_state] - V[-dev_t][start_state]
            td_residual_array.append(td_residual)
        td_residual_array_sum = list(accumulate(td_residual_array))
        for dev_t, dev_a, dev_imp_weight, dev_ex, transition_ex in zip(
                self.dev_t, self.dev_a, self.dev_imp_weight, self.dev_ex, self.transition_ex):
            start_state = [float(x.split(':')[1]) for x in transition_ex.replace('|', '').strip().split()[:-1]][:16].index(1.0)
            end_state = [float(x.split(':')[1]) for x in transition_ex.replace('|', '').strip().split()[:-1]][16:].index(1.0)
            advantage = Q[-dev_t][start_state, dev_a] - V[-dev_t][start_state]
            td_residual = costs_function[start_state, dev_a] + final_state.example.gamma * V[-dev_t-1][end_state] - V[-dev_t][start_state]
            c_formula = loss0 - V[-1][3] - (td_residual_array_sum[dev_t-1] - td_residual_array[dev_t-1])
            print('diff: ', td_residual - c_formula)
            print('TD Residual: ', td_residual)
#            print('TD Residual array:', td_residual_array[dev_t-1])
            print('C Formula: ', c_formula)
            print('Advantage: ', advantage)
            print('===================================')
#            pred_vd = self.pred_act_cost[dev_t-1]
#            residual_loss = loss0 - initial_state_value - (prefix_sum[dev_t-1] - self.pred_act_cost[dev_t-1])
#            vd_sq_loss = (residual_loss - pred_vd) ** 2
#            self.total_vd_sq_loss += vd_sq_loss
#            transition_example = str(residual_loss) + transition_ex
#            self.vw_vd_regressor.learn(transition_example)
#            bandit_loss = residual_loss
#            bandit_loss = advantage
            bandit_loss = td_residual
#            bandit_loss = c_formula
            self.policy.update(dev_a, bandit_loss, dev_imp_weight, dev_ex)
        self.vw_ref_critic.learn(initial_state_ex)
        self.t, self.dev_t, self.dev_a, self.dev_imp_weight, self.pred_act_cost, self.dev_ex, self.transition_ex = [None] * 7
        return loss0
