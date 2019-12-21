from itertools import accumulate

import numpy as np
import torch
from torch.autograd import Variable as Var

import macarico
import macarico.util
from macarico.annealing import NoAnnealing, stochastic
from macarico.lts.lols import BanditLOLS, LOLS


class VdReslope(BanditLOLS):

    def __init__(self, reference, policy, ref_critic=None, vd_regressor=None, actor=None, residual_loss_clip_fn=None,
                 p_ref=stochastic(NoAnnealing(0)), update_method=BanditLOLS.LEARN_DR,
                 exploration=BanditLOLS.EXPLORE_BOLTZMANN, explore=1.0, mixture=LOLS.MIX_PER_ROLL, temperature=0.1,
                 save_log=False, writer=None, attach_time = True):
        super(VdReslope, self).__init__(policy=policy, reference=reference, exploration=exploration, mixture=mixture,
                                        update_method=update_method, temperature=temperature)
        self.reference = reference
        self.policy = policy
        self.ref_critic = ref_critic
        self.vd_regressor = vd_regressor
        self.update_method = update_method
        self.exploration = exploration
        self.temperature = temperature
        assert self.update_method in range(BanditLOLS._LEARN_MAX), \
            'unknown update_method, must be one of BanditLOLS.LEARN_*'
        assert self.exploration in range(BanditLOLS._EXPLORE_MAX), \
            'unknown exploration, must be one of BanditLOLS.EXPLORE_*'
        # TODO assert correct exploration type with boltzmann
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
        self.save_log = save_log
        # TODO why 200?
        self.per_step_count = np.zeros(200)
        self.counter = 0
        if save_log:
            self.writer = writer
            self.action_count = 0
            self.critic_losses = []
            self.td_losses = []
        self.actor = actor
        self.residual_loss_clip_fn = residual_loss_clip_fn

    def forward(self, state):
        self.per_step_count[self.t] += 1
        if self.t is None or self.t == []:
            self.T = state.horizon()
            self.init_state = self.actor(state).data
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
            if self.attach_time:
                curr_loss = torch.Tensor([[state.loss(self.t-2), self.t]])
            else:
                curr_loss = torch.Tensor([[state.loss(self.t-2)]])
            transition_tuple = torch.cat([self.prev_state, self.actor(state).data, curr_loss], dim=1)
            pred_vd = self.vd_regressor(transition_tuple)
            self.pred_vd.append(pred_vd)
            val = self.residual_loss_clip_fn(pred_vd.data.numpy())
            # val = pred_vd.data.numpy()
            self.pred_act_cost.append(val)
        self.prev_state = self.actor(state).data

        a_pol = self.policy(state)
        a_costs = self.policy.predict_costs(state)
        if self.reference is not None:
            a_ref = self.reference(state)
            return a_ref
        # exploit
        if not self.explore():
            a = a_pol
        else:
            dev_a, iw = self.do_exploration(a_costs, state.actions)
            a = dev_a if isinstance(dev_a, int) else dev_a.data[0,0]
            self.dev_t.append(self.t)
            self.dev_a.append(a)
            self.dev_actions.append(list(state.actions)[:])
            self.dev_imp_weight.append(iw)
            self.dev_costs.append(a_costs)
        return a

    def get_objective(self, loss0, final_state=None):
        loss0 = float(loss0)
        self.counter += 1
        total_loss_var = 0.
        if self.attach_time:
            terminal_loss = torch.Tensor([[final_state.loss(self.t - 1), self.t]])
        else:
            terminal_loss = torch.Tensor([[final_state.loss(self.t - 1)]])
        transition_tuple = torch.cat([self.prev_state, self.actor(final_state).data, terminal_loss], dim=1)
        # pred_vd = self.vd_regressor.predict_costs(final_state)
        pred_vd = self.vd_regressor(transition_tuple)
        self.pred_vd.append(pred_vd)
        val = self.residual_loss_clip_fn(pred_vd.data.numpy())
        # val = pred_vd.data.numpy()
        self.pred_act_cost.append(val)
        pred_value = self.ref_critic(self.init_state)
        if self.reference is None:
            loss = self.ref_critic.update(pred_value, torch.Tensor([[loss0]]))
            total_loss_var += loss
        if self.save_log:
            self.writer.add_scalar('trajectory_loss', loss0, self.counter)
            self.writer.add_scalar('predicted_loss', pred_value, self.counter)
            self.critic_losses.append(loss.data.numpy())
            self.writer.add_scalar('critic_loss', np.mean(self.critic_losses[-50:]), self.counter)
        if self.dev_t is not None:
            prefix_sum = list(accumulate(self.pred_act_cost))
            for dev_t, dev_a, dev_actions, dev_imp_weight, dev_costs in zip(self.dev_t, self.dev_a, self.dev_actions,
                                                                            self.dev_imp_weight, self.dev_costs):
                if dev_costs is None or dev_imp_weight == 0.:
                    continue
                dev_costs_data = dev_costs.data if isinstance(dev_costs, Var) else \
                    dev_costs.data() if isinstance(dev_costs, macarico.policies.bootstrap.BootstrapCost) else \
                        None
                assert dev_costs_data is not None

                residual_loss = loss0 - (prefix_sum[dev_t-1] - self.pred_act_cost[dev_t-1])
                # residual_loss = loss0 - pred_value.data.numpy() - (prefix_sum[dev_t-1] - self.pred_act_cost[dev_t-1])
                # residual_loss = self.residual_loss_clip_fn(residual_loss)
                tdiff_loss = self.vd_regressor.update(self.pred_vd[dev_t-1], torch.Tensor(residual_loss))
                total_loss_var += tdiff_loss

                if self.save_log:
                    self.writer.add_scalar('TDIFF-loss/' + f'{dev_t-1}', tdiff_loss.data.numpy(), self.per_step_count[dev_t-1])
                    self.writer.add_scalar('TDIFF-predicted_tdiff/'+ f'{dev_t-1}', self.pred_act_cost[dev_t-1], self.per_step_count[dev_t-1])
                    self.writer.add_scalar('TDIFF-residual_loss/'+f'{dev_t-1}', residual_loss, self.per_step_count[dev_t-1])

                reslope_loss = loss0 - (np.sum(self.pred_act_cost) - self.pred_act_cost[dev_t - 1])

                self.build_truth_vector(residual_loss, dev_a, dev_imp_weight, dev_costs_data)
                importance_weight = 1
                if self.update_method in [BanditLOLS.LEARN_MTR, BanditLOLS.LEARN_MTR_ADVANTAGE]:
                    dev_actions = [dev_a if isinstance(dev_a, int) else dev_a.data[0,0]]
                    importance_weight = dev_imp_weight
                loss_var = self.policy.update(dev_costs, self.truth, dev_actions)
                loss_var *= importance_weight
                total_loss_var += loss_var

                a = dev_a if isinstance(dev_a, int) else dev_a.data[0,0]
                self.squared_loss = (loss0 - dev_costs_data[a]) ** 2
        # Update VD regressor using all timesteps
        self.use_ref.step()
        self.t, self.dev_t, self.dev_a, self.dev_actions, self.dev_imp_weight, self.dev_costs, self.pred_vd, self.pred_act_cost, self.rollout = [None] * 9
        return total_loss_var