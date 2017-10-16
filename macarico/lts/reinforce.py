from __future__ import division

import sys
import random
import dynet as dy
#import torch
#import torch.nn.functional as F
#from torch import autograd
#from torch.autograd import Variable

import macarico


class Reinforce(macarico.Learner):
    "REINFORCE with a scalar baseline function."

    def __init__(self, policy, baseline, max_deviations=None, uniform=False, temperature=1.0):
        self.trajectory = []
        self.baseline = baseline
        self.policy = policy
        self.max_deviations = max_deviations
        self.t = None
        self.dev_t = None
        self.temperature = 1000 if uniform else temperature
        super(Reinforce, self).__init__()

    def update(self, loss):
        if len(self.trajectory) > 0:
            b = self.baseline()
            total_loss = 0
            for a, p_a in self.trajectory:
                total_loss += (loss - b) * dy.log(p_a)
            self.baseline.update(loss)
            total_loss.forward()
            total_loss.backward()
        #torch.autograd.backward(self.trajectory[:], [None]*len(self.trajectory))

    def __call__(self, state):
        if self.t is None:
            self.t = 0
            if self.max_deviations is not None:
                t_list = range(1, state.T+1)
                random.shuffle(t_list)
                self.dev_t = set(t_list[:self.max_deviations])

        self.t += 1

        if self.max_deviations is None or self.t in self.dev_t:
            action, p_action = self.policy.stochastic_with_probability(state, temperature=self.temperature)
            # log actions (and values for actor critic) taken along current trajectory
            self.trajectory.append((action, p_action))
        else:
            action = self.policy.greedy(state)
        return action

# TODO: scalar baseline should be an instance of this class with one constant feature.
class LinearValueFn(object):
    """
    Linear value function regressor.
    """

    def __init__(self, dy_model, dim):
        self.dy_model = dy_model
        self._w = dy_model.add_parameters((1, dim))
        self._b = dy_model.add_parameters(1)

#    @profile
    def __call__(self, feats):
        w = dy.parameter(self._w)
        b = dy.parameter(self._b)
        return dy.affine_transform([b, w, feats])


class AdvantageActorCritic(macarico.Learner):

    def __init__(self, policy, state_baseline, vfa_multiplier=1.0, temperature=1.0):
        self.policy = policy
        self.baseline = state_baseline
        self.values = []
        self.trajectory = []
        self.vfa_multiplier = vfa_multiplier
        self.temperature = temperature
        super(AdvantageActorCritic, self).__init__()

#    @profile
    def update(self, loss):
        if len(self.trajectory) > 0:
            total_loss = 0.0
            for (a, p_a), v, in zip(self.trajectory, self.values):
                # a.reinforce(v.data.view(1)[0] - loss)
                b = v.npvalue()[0]
                total_loss += (loss - b) * dy.log(p_a)

                # TODO: loss should live in the VFA, similar to policy
                total_loss += self.vfa_multiplier * dy.huber_distance(v, dy.inputTensor([loss]))

            total_loss.forward()
            total_loss.backward()

#    @profile
    def __call__(self, state):
        action, p_action = self.policy.stochastic_with_probability(state, temperature=self.temperature)
        value = self.baseline(self.policy.features(state))

        # log actions and values taken along current trajectory
        self.trajectory.append((action, p_action))
        self.values.append(value)

        return action
