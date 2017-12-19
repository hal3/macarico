from __future__ import division

from macarico.policies.linear import LinearPolicy
from macarico import Policy
from macarico import util
import numpy as np
import dynet as dy


def actions_to_probs(actions, n_actions):
    probs = np.zeros(n_actions)
    bag_size = len(actions)
    prob = 1. / bag_size
    for action_set in actions:
        for action in action_set:
            probs[action] += prob / len(action_set)
    return probs


class EnsembleCost:
    def __init__(self, costs):
        self.costs = costs

    def softmax(self):
        return dy.softmax(dy.average(self.costs)).npvalue()

    def npvalue(self):
        return dy.average(self.costs).npvalue()

    def __getitem__(self, idx):
        return dy.average(self.costs)[idx]

    def __neg__(self):
        return dy.average(self.costs).__neg__()

    def argmin(self):
        return dy.average(self.costs).argmin()


# Constructs a policy bag of linear policies, number of policies =
# len(features_bag)
def build_policy_bag(dy_model, features_bag, n_actions, loss_fn, n_layers,
                     hidden_dim):
    return [LinearPolicy(dy_model, features, n_actions, loss_fn=loss_fn,
                         n_layers=n_layers, hidden_dim=hidden_dim)
            for features in features_bag]


def delegate(params, functions):
    total_loss = 0.0
    functions_params_pairs = zip(functions, params)
    for idx, (loss_fn, params) in enumerate(functions_params_pairs):
        loss_i = loss_fn(*params)
        total_loss = total_loss + loss_i
    return total_loss


class EnsemblePolicy(Policy):
    """
        Ensemble policy
    """

    def __init__(self, dy_model, features_bag, n_actions, loss_fn='squared',
                 n_layers=1, hidden_dim=50):
        self.n_actions = n_actions
        self.bag_size = len(features_bag)
        self.policy_bag = build_policy_bag(dy_model, features_bag, n_actions,
                                           loss_fn, n_layers, hidden_dim)

    def __call__(self, state, deviate_to=None):
        ensemble_cost = self.predict_costs(state, deviate_to)
        action_probs = ensemble_cost.softmax()
        action, _ = util.sample_from_np_probs(action_probs)
        return action

    def predict_costs(self, state, deviate_to=None):
        all_costs = [policy.predict_costs(state, deviate_to)
                     for policy in self.policy_bag]
        return EnsembleCost(all_costs)

    def greedy(self, state, pred_costs=None, deviate_to=None):
        actions = [[policy.greedy(state, pred_costs=p_costs,
                                  deviate_to=deviate_to)]
                   for policy, p_costs in zip(self.policy_bag, pred_costs)]
        action_probs = actions_to_probs(actions, self.n_actions)
        action, prob = util.sample_from_np_probs(action_probs)
        return action

    def forward(self, state, ref):
        params = [(state, ref) for i in range(self.bag_size)]
        fns = [policy.forward for policy in self.policy_bag]
        return delegate(params, fns)

    def forward_partial_complete(self, costs, truth, acts):
        params = [(costs.costs[i], truth, acts) for i in range(self.bag_size)]
        loss_fns = [p.forward_partial_complete for p in self.policy_bag]
        return delegate(params, loss_fns)
