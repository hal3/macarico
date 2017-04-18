from __future__ import division

from maximum_likelihood import MaximumLikelihood
from annealing import ExponentialAnnealing
import random

# we derive DAgger from MaximumLikelihood because the only difference
# is how we do roll-ins
class DAgger(MaximumLikelihood):
    def __init__(self, p_rollin_ref=ExponentialAnnealing(0.99)):
        super(DAgger, self).__init__()

        # remember how many examples we've trained on, and the
        # computation for p(ref) on rollin
        self.n_examples   = 0.
        self.p_rollin_ref = p_rollin_ref

    def train(self, task, input):
        # increment number of training examples
        self.n_examples += 1.
        # train is identical to MaximumLikelihood.train
        return super(DAgger, self).train(task, input)

    def act(self, state, a_ref=None):
        # the objective function for DAgger is identical to that for
        # MaximumLikelihood, so just do that.
        super(DAgger, self).act(state, a_ref)

        # in DAgger, with probability p_rollin_ref we use the
        # reference; and with probability 1-p_rollin_ref we act
        # greedily according to the current policy
        use_ref = random.random() <= self.p_rollin_ref(self.current_pass) #self.n_examples)
        return a_ref if use_ref else self.task.act_greedy(state)
