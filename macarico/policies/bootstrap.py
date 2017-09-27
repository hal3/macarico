from __future__ import division

from macarico.policies.linear import LinearPolicy
from macarico import Policy
from macarico import util
import numpy as np
import dynet as dy


# Randomize over predictions from a base set of predictors
def bootstrap_probabilities(n_actions, bag_size, policy_bag, state, deviate_to):
    preds = np.zeros(n_actions)
    prob = 1. / bag_size
    for policy in policy_bag:
        a = policy(state, deviate_to)
        preds[a] += prob
    return preds


# Constructs a policy bag of linear policies, number of policies =
# len(features_bag)
def build_policy_bag(dy_model, features_bag, n_actions, loss_fn):
    # TODO do we need to deep copy the dy_model?
    return [LinearPolicy(dy_model, features, n_actions, loss_fn)
            for features in features_bag]


class BootstrapPolicy(Policy):
    """
        Bootstrapping policy
        TODO: how can we train this policy?
    """

    def __init__(self, dy_model, features_bag, n_actions, loss_fn='squared'):
        self.n_actions = n_actions
        self.bag_size = len(features_bag)
        self.policy_bag = build_policy_bag(dy_model, features_bag, n_actions,
                                           loss_fn)

    def __call__(self, state, deviate_to=None):
        action_probs = bootstrap_probabilities(
            self.n_actions, self.bag_size, self.policy_bag, state, deviate_to)
        print('Action probabilities: ', action_probs)
        return util.sample_from_np_probs(action_probs)

    def predict_costs(self, state, deviate_to=None):
        all_costs = [policy.predict_costs(state, deviate_to) for policy in self.policy_bag]
        return dy.average(all_costs)

