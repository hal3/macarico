from __future__ import division

from macarico import Policy
import numpy as np

# Randomize over predictions from a base set of predictors
def bootstrap_probabilities(num_actions, bag_size, policy_bag, state):
    preds = np.zeros(num_actions)
    prob = 1. / bag_size;
    for policy in policy_bag:
        # get the prediction
        a = policy(state)
        # update probability scores
        preds[a] += prob
    return preds


# Constructs a policy bag of linear policies, number of policies = bag_size
def build_policy_bag(bag_size):
    return [LinearPolicy() for i in range(bag_size)]

class BootstrapPolicy(Policy):
    """
        Bootstrapping policy
        TODO: how can we train this policy?
    """

    def __init__(self, dy_model, features, n_actions, bag_size=16, loss_fn='squared'):
        self.policy_bag = None
        return None

    def __call__(self, state, deviate_to=None):
        return None

if __name__ == '__main__':
    num_actions = 10
    bag_size = 16
    policy_bag = []
    state = None
    print(bootstrap_probabilities(num_actions, bag_size, policy_bag, state))
