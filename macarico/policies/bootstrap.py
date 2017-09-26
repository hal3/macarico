from __future__ import division

from macarico import Policy


class BootstrapPolicy(Policy):
    """Bootstrapping policy
    """

    def __init__(self, dy_model, features, n_actions, loss_fn='squared'):
        return None

    def __call__(self, state, deviate_to=None):
        return None
