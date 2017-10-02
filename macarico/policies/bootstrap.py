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
    for action in actions:
        probs[action] += prob
    return probs


# Randomize over predictions from a base set of predictors
def bootstrap_probabilities(n_actions, policy_bag, state, deviate_to):
    actions = [policy(state, deviate_to) for policy in policy_bag]
    probs = actions_to_probs(actions, n_actions)
    return probs


class BootstrapCost:
    def __init__(self, costs, greedy_predict=True):
        self.costs = costs
        self.greedy_predict = greedy_predict

    def npvalue(self):
        if self.greedy_predict:
            return self.costs[0].npvalue()
        else:
            return dy.average(self.costs).npvalue()

    def get_probs(self):
        assert(len(self.costs) > 0)
        n_actions = len(self.costs[0].npvalue())
        actions = [c.npvalue().argmin() for c in self.costs]
        return actions_to_probs(actions, n_actions)

    def __getitem__(self, idx):
        if self.greedy_predict:
            return self.costs[0][idx]
        else:
            return dy.average(self.costs)[idx]

    def __neg__(self):
        if self.greedy_predict:
            return self.costs[0].__neg__()
        else:
            return dy.average(self.costs).__neg__()

# Sampling from Poisson with rate 1
def poisson_sample():
    temp = np.random.random()
    if(temp <= 0.3678794411714423215955):
        return 0
    if(temp <= 0.735758882342884643191):
        return 1
    if(temp <= 0.919698602928605803989):
        return 2
    if(temp <= 0.9810118431238461909214):
        return 3
    if(temp <= 0.9963401531726562876545):
        return 4
    if(temp <= 0.9994058151824183070012):
        return 5
    if(temp <= 0.9999167588507119768923):
        return 6
    if(temp <= 0.9999897508033253583053):
        return 7
    if(temp <= 0.9999988747974020309819):
        return 8
    if(temp <= 0.9999998885745216612793):
        return 9
    if(temp <= 0.9999999899522336243091):
        return 10
    if(temp <= 0.9999999991683892573118):
        return 11
    if(temp <= 0.9999999999364022267287):
        return 12
    if(temp <= 0.999999999995480147453):
        return 13
    if(temp <= 0.9999999999996999989333):
        return 14
    if(temp <= 0.9999999999999813223654):
        return 15
    if(temp <= 0.9999999999999989050799):
        return 16
    if(temp <= 0.9999999999999999393572):
        return 17
    if(temp <= 0.999999999999999996817):
        return 18
    if(temp <= 0.9999999999999999998412):
        return 19
    return 20


# Constructs a policy bag of linear policies, number of policies =
# len(features_bag)
def build_policy_bag(dy_model, features_bag, n_actions, loss_fn):
    return [LinearPolicy(dy_model, features, n_actions, loss_fn)
            for features in features_bag]

class BootstrapPolicy(Policy):
    """
        Bootstrapping policy
    """

    def __init__(self, dy_model, features_bag, n_actions, loss_fn='squared',
                 greedy_predict=True, greedy_update=True):
        self.n_actions = n_actions
        self.bag_size = len(features_bag)
        self.policy_bag = build_policy_bag(dy_model, features_bag, n_actions,
                                           loss_fn)
        self.greedy_predict = greedy_predict
        self.greedy_update = greedy_update

    def __call__(self, state, deviate_to=None):
        action_probs = bootstrap_probabilities(self.n_actions, self.policy_bag,
                                               state, deviate_to)
        if self.greedy_predict:
            return self.policy_bag[0](state, deviate_to)
        else:
            action, prob = util.sample_from_np_probs(action_probs)
            return action

    def predict_costs(self, state, deviate_to=None):
        all_costs = [policy.predict_costs(state, deviate_to)
                     for policy in self.policy_bag]
        return BootstrapCost(all_costs, self.greedy_predict)

    def forward_partial_complete(self, all_costs, truth, actions):
        total_loss = 0.0
        policies_costs_pairs = zip(self.policy_bag, all_costs.costs)
        for idx, (policy, pred_costs) in enumerate(policies_costs_pairs):
            loss_i = policy.forward_partial_complete(pred_costs, truth, actions)
            if self.greedy_update and idx == 0:
                count_i = 1
            else:
                count_i = poisson_sample()
            total_loss = total_loss + count_i * loss_i
        return total_loss
