from __future__ import division

import numpy as np
import math
import macarico


class Example(object):
    """
    >>> e = Example('abcdef', 'ABCDEF', 7)
    >>> env = e.mk_env()
    >>> env.run_episode(env.reference())
    ['A', 'B', 'C', 'D', 'E', 'F']
    >>> env.loss()
    0.0
    >>> env = e.mk_env()
    >>> env.run_episode(lambda s: s.tokens[s.n].upper() if s.n % 2 else '_')
    ['_', 'B', '_', 'D', '_', 'F']
    >>> env.loss()
    0.5
    """

    def __init__(self, tokens, labels, n_labels):
        self.tokens = tokens
        self.labels = labels
        self.n_labels = n_labels

    def mk_env(self):
        return SequenceLabeling(self, self.n_labels)

    def __str__(self):
        return ' '.join(map(str, self.labels))


class SequenceLabeling(macarico.Env):
    """Basic sequence labeling environment (input and output sequences have the same
    length). Loss is evaluated with Hamming distance, which has an optimal
    reference policy.

    """

    def __init__(self, example, n_labels):
        self.example = example
        self.N = len(example.tokens)
        self.T = self.N
        self.n = None
        self.t = None
        self.prev_action = None          # previous action
        self.output = []
        self.tokens = example.tokens
        self.actions = set(range(n_labels))
        super(SequenceLabeling, self).__init__(n_labels)

    def rewind(self):
        self.n = None
        self.t = None
        self.prev_action = None          # previous action
        self.output = []

    def run_episode(self, policy):
        self.output = []
        for self.n in xrange(self.N):
            self.t = self.n
            a = policy(self)
            self.output.append(a)
        return self.output


class LossTrajectory:
    def __init__(self, trajectory, combiner=sum):
        self.trajectory = trajectory
        self.combiner = combiner

    def __sub__(self, other):
        return self.combiner(self.trajectory) - other

    def __iadd__(self, other):
        return self.combiner(self.trajectory) + other

    def __radd__(self, other):
        return self.combiner(self.trajectory) + other

    def __len__(self):
        return len(self.trajectory)

    def __getitem__(self, index):
        return self.trajectory[index]


class HammingLossReference(macarico.Reference):
    def __init__(self):
        pass

    def __call__(self, state):
        return state.example.labels[state.n]

    def set_min_costs_to_go(self, state, cost_vector):
        cost_vector *= 0
        cost_vector += 1
        cost_vector[state.example.labels[state.n]] = 0.

class HammingLoss(macarico.Loss):
    def __init__(self):
        super(HammingLoss, self).__init__('hamming')

    def evaluate(self, ex, state):
        assert len(state.output) == len(ex.labels), 'can only evaluate loss at final state'
        loss_trajectory = [int(y != p) for p,y in zip(state.output, ex.labels)]
        return LossTrajectory(loss_trajectory)


class TimeSensitiveHammingLoss(macarico.Loss):
    def __init__(self):
        super(TimeSensitiveHammingLoss, self).__init__('time_sensitive_hamming')

    def evaluate(self, ex, state):
        assert len(state.output) == len(ex.labels), \
            'can only evaluate los at final state'
        loss_trajectory = [(t + 1) * int(y != p) for t, (p, y)
                           in enumerate(zip(state.output, ex.labels))]
        return LossTrajectory(loss_trajectory)


class DistanceSensitiveHammingLoss(macarico.Loss):
    def __init__(self):
        super(DistanceSensitiveHammingLoss, self).__init__('distance_sensitive_hamming')

    def evaluate(self, ex, state):
        assert len(state.output) == len(ex.labels), \
            'can only evaluate los at final state'
        loss_trajectory = [float(np.abs(y - p)) for t, (p, y)
                           in enumerate(zip(state.output, ex.labels))]
        return LossTrajectory(loss_trajectory)


def l2_combiner(loss_trajectory):
    return math.sqrt(sum(loss_trajectory))


def product_combiner(loss_trajectory):
    product = 1.0
    for loss in loss_trajectory:
        product *= loss
    return product


class EuclideanHammingLoss(macarico.Loss):
    def __init__(self):
        super(EuclideanHammingLoss, self).__init__('euclidean_hamming')

    def evaluate(self, ex, state):
        assert len(state.output) == len(ex.labels), \
            'can only evaluate los at final state'
        loss_trajectory = [int(y != p) for p, y in zip(state.output, ex.labels)]
        return LossTrajectory(loss_trajectory, combiner=l2_combiner)


class ProductHammingLoss(macarico.Loss):
    def __init__(self):
        super(ProductHammingLoss, self).__init__('product_hamming')

    def evaluate(self, ex, state):
        assert len(state.output) == len(ex.labels), \
            'can only evaluate los at final state'
        loss_trajectory = [int(y != p) for p, y in zip(state.output, ex.labels)]
        return LossTrajectory(loss_trajectory, combiner=product_combiner)
