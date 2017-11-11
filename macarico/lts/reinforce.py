from __future__ import division

import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var

from macarico.annealing import EWMA
#import torch
#import torch.nn.functional as F
#from torch import autograd
#from torch.autograd import Variable

import macarico


class Reinforce(macarico.Learner):
    "REINFORCE with a scalar baseline function."

    def __init__(self, policy, baseline=None, max_deviations=None, uniform=False, temperature=1.0):
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
            b = 0 if self.baseline is None else self.baseline()
            total_loss = 0
            for a, p_a in self.trajectory:
                total_loss += (loss - b) * dy.log(p_a)
            if self.baseline is not None:
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

    def __init__(self,dim):
        
        self._w = dy_model.add_parameters((1, dim), init=dy.NumpyInitializer(torch.zeros((1,dim))))
        self._b = dy_model.add_parameters(1, init=dy.NumpyInitializer(torch.zeros(1)))

#    @profile
    def __call__(self, feats):
        w = dy.parameter(self._w)
        b = dy.parameter(self._b)
        return dy.affine_transform([b, w, feats])


stupid_baseline = EWMA(0.8)

class AdvantageActorCritic(macarico.Learner):

    def __init__(self, policy, state_baseline, disconnect_values=True, vfa_multiplier=1.0, temperature=1.0):
        self.policy = policy
        self.baseline = state_baseline
        self.values = []
        self.trajectory = []
        self.vfa_multiplier = vfa_multiplier
        self.temperature = temperature
        self.disconnect_values = disconnect_values
        super(AdvantageActorCritic, self).__init__()

#    @profile
    def update(self, loss):
        global stupid_baseline
        if len(self.trajectory) > 0:
            b2 = stupid_baseline()
            total_loss = 0.0
            for (a, p_a), v in zip(self.trajectory, self.values):
                # a.reinforce(v.data.view(1)[0] - loss)
                b = v.data[0]
                #b2 = (b + b2) / 2
                #b3 = 0.5 * b2 + 0.5 * b
                b3 = b
                #if np.random.random() < 0.1: print b, b2, b3, loss
                total_loss += (loss - b3) * dy.log(p_a)

                # TODO: loss should live in the VFA, similar to policy
                total_loss += self.vfa_multiplier * dy.huber_distance(v, dy.inputTensor([loss]))
                #total_loss += self.vfa_multiplier * (v-loss) * (v-loss)

            total_loss.forward()
            total_loss.backward()
            stupid_baseline.update(loss)

#    @profile
    def __call__(self, state):
        action, p_action = self.policy.stochastic_with_probability(state, temperature=self.temperature)
        feats = self.policy.features(state)
        if self.disconnect_values:
            feats = dy.inputTensor(feats.data)
        #feats = dy.inputTensor([0])
        value = self.baseline(feats)

        # log actions and values taken along current trajectory
        self.trajectory.append((action, p_action))
        self.values.append(value)

        return action
