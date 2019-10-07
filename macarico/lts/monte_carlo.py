import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable as Var

import macarico
import macarico.policies.bootstrap
import macarico.policies.costeval
import macarico.util
from macarico.annealing import NoAnnealing, stochastic
from macarico.lts.lols import BanditLOLS, LOLS, EpisodeRunner


class MonteCarlo(macarico.Learner):
    def __init__(self, policy, reference=None, p_rollin_ref=NoAnnealing(0), p_rollout_ref=NoAnnealing(0.5),
                 update_method=BanditLOLS.LEARN_MTR, exploration=BanditLOLS.EXPLORE_BOLTZMANN,
                 p_explore=NoAnnealing(1.0), mixture=LOLS.MIX_PER_ROLL, temperature=1.0, is_episodic=True):
        macarico.Learner.__init__(self)
        if reference is None:
            reference = lambda s: np.random.choice(list(s.actions))
        self.policy = policy
        self.reference = reference
        self.rollin_ref = stochastic(p_rollin_ref)
        self.rollout_ref = stochastic(p_rollout_ref)
        self.update_method = update_method
        self.exploration = exploration
        self.explore = stochastic(p_explore)
        self.mixture = mixture
        self.temperature = temperature
        self.episodic = is_episodic

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
        self.disallow = torch.zeros(self.policy.n_actions)
        self.truth = torch.zeros(self.policy.n_actions)

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
            return int(np.random.choice(list(dev_actions))), len(dev_actions)
        if self.exploration in [BanditLOLS.EXPLORE_BOLTZMANN, BanditLOLS.EXPLORE_BOLTZMANN_BIASED]:
            if len(dev_actions) != self.policy.n_actions:
                self.disallow.zero_()
                for i in range(self.policy.n_actions):
                    if i not in dev_actions:
                        self.disallow[i] = 1e10
                costs += Var(self.disallow, requires_grad=False)
            probs = F.softmax(- costs / self.temperature, dim=0)
            a, p = macarico.util.sample_from_probs(probs)
            p = p.data.item()
            if self.exploration == BanditLOLS.EXPLORE_BOLTZMANN_BIASED:
                p = max(p, 1e-4)
            return a, 1 / p
        if self.exploration == BanditLOLS.EXPLORE_BOOTSTRAP:
            if self.explore():
                return int(np.random.choice(list(dev_actions))), len(dev_actions)
            else:
                assert isinstance(self.policy, macarico.policies.bootstrap.BootstrapPolicy) or \
                       isinstance(self.policy, macarico.policies.costeval.CostEvalPolicy)
                # TODO assert costs are bootstrap costs
                probs = costs.get_probs(dev_actions)
                a, p = macarico.util.sample_from_np_probs(probs)
                return a, 1 / p
        assert False, 'unknown exploration strategy'

    def get_objective(self, loss, final_state=None):
        loss = float(loss)

        obj = 0.
        # if self.episodic:
        #     returns = loss0*np.ones(len(self.dev_a))
        # else:
        #     returns = np.zeros(len(loss0))
        #     for i in reversed(range(len(loss0))):
        #         for j in range
        if self.dev_a is not None:
            if isinstance(self.dev_a, list):
                for dev_a, dev_costs, dev_imp_weight in zip(self.dev_a, self.dev_costs, self.dev_imp_weight):
                    dev_costs_data = dev_costs.data if isinstance(self.dev_costs, Var) else None
                    self.build_truth_vector(loss, dev_a, dev_imp_weight, dev_costs_data)
                    importance_weight = 1.0
                    if self.update_method in [BanditLOLS.LEARN_MTR, BanditLOLS.LEARN_MTR_ADVANTAGE]:
                        self.dev_actions = [dev_a if isinstance(dev_a, int) else dev_a.data[0,0]]
                        importance_weight = dev_imp_weight
                    loss_var = self.policy.update(dev_costs, self.truth, self.dev_actions)
                    loss_var *= importance_weight
                    obj = loss_var
            else:
                # TODO support bootstrap costs
                dev_costs_data = self.dev_costs.data if isinstance(self.dev_costs, Var) else None
                self.build_truth_vector(final_state.loss_to_go(self.dev_t), self.dev_a, self.dev_imp_weight,
                                        dev_costs_data)
                importance_weight = 1.0
                if self.update_method in [BanditLOLS.LEARN_MTR, BanditLOLS.LEARN_MTR_ADVANTAGE]:
                    self.dev_actions = [self.dev_a if isinstance(self.dev_a, int) else self.dev_a[0]]
                    importance_weight = self.dev_imp_weight
                loss_var = self.policy.update(self.dev_costs, self.truth, self.dev_actions)
                loss_var *= importance_weight
                obj = loss_var

        self.explore.step()
        self.rollin_ref.step()
        self.rollout_ref.step()

        self.t, self.dev_t, self.dev_a, self.dev_actions, self.dev_imp_weight, self.dev_costs, self.rollout = [None] * 7
        return obj

    def build_truth_vector(self, loss, a, imp_weight, dev_costs_data):
        self.truth.zero_()
        if not isinstance(a, int): a = a.data[0,0]
        if self.update_method == BanditLOLS.LEARN_BIASED:
            self.truth[a] = loss
        elif self.update_method == BanditLOLS.LEARN_IPS:
            self.truth[a] = loss * imp_weight
        elif self.update_method == BanditLOLS.LEARN_DR:
            self.truth += dev_costs_data # now costs = \hat c
            self.truth[a] = dev_costs_data[a] + imp_weight * (loss - dev_costs_data[a])
        elif self.update_method == BanditLOLS.LEARN_MTR:
            self.truth[a] = float(loss)
        elif self.update_method == BanditLOLS.LEARN_MTR_ADVANTAGE:
            self.truth[a] = loss - dev_costs_data.min()
        else:
            assert False, self.update_method

