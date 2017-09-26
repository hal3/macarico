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


class BootstrapPolicy(Policy):
    """Bootstrapping policy
    """

    def __init__(self, dy_model, features, n_actions, loss_fn='squared'):
        return None

    def __call__(self, state, deviate_to=None):
        return None

if __name__ == '__main__':
    num_actions = 10
    bag_size = 16
    policy_bag = []
    state = None
    print(bootstrap_probabilities(num_actions, bag_size, policy_bag, state))
