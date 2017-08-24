from __future__ import division

import random
import dynet as dy
#import torch
#import torch.nn.functional as F
#from torch import autograd
#from torch.autograd import Variable

import macarico


class Reinforce(macarico.Learner):
    "REINFORCE with a scalar baseline function."

    def __init__(self, policy, baseline):
        self.trajectory = []
        self.baseline = baseline
        self.policy = policy
        super(Reinforce, self).__init__()

    def update(self, loss):
        b = self.baseline()
        total_loss = 0
        for a, p_a in self.trajectory:
            total_loss += (loss - b) * dy.log(p_a)
            #total_loss -= ((b - loss) / p_a.npvalue()[0]) * p_a
            # (b-loss) * D log(p(a|s))
            # (b-loss) * [ D p(a|s) ] / p(a|s)
        self.baseline.update(loss)
        total_loss.forward()
        total_loss.backward()
        #torch.autograd.backward(self.trajectory[:], [None]*len(self.trajectory))

    def __call__(self, state):
        action, p_action = self.policy.stochastic_with_probability(state)
        # log actions (and values for actor critic) taken along current trajectory
        self.trajectory.append((action, p_action))
        return action

# TODO: scalar baseline should be an instance of this class with one constant feature.
class LinearValueFn(object):
    """
    Linear value function regressor.
    """

    def __init__(self, dy_model, features):
        self.dy_model = dy_model
        self._w = dy_model.add_parameters((1, features.dim))
        self._b = dy_model.add_parameters(1)
        self.features = features

#    @profile
    def __call__(self, state):
        w = dy.parameter(self._w)
        b = dy.parameter(self._b)
        return dy.affine_transform([b, w, self.features(state)])
        #return self._predict(self.features(state))


class AdvantageActorCritic(macarico.Learner):

    def __init__(self, policy, state_baseline, gamma=1.0):
        self.policy = policy
        self.baseline = state_baseline
        self.values = []
        self.trajectory = []
        self.gamma = gamma
        super(AdvantageActorCritic, self).__init__()

#    @profile
    def update(self, loss):

        rewards = [loss] * len(self.trajectory)
#        rewards = []
#        R = 0.0
#        for r in [loss]*len(self.trajectory): #self.saved_rewards[::-1]:    # reverse
#            R = r + self.gamma * R
#            rewards.append(R)
#        rewards = torch.Tensor(rewards[::-1]) # un-reverse

        # this step is a little weird. got it from example in torch repo.
#        rewards = (rewards - rewards.mean()) / rewards.std()

        total_loss = 0.0
        for (a, p_a), v, r in zip(self.trajectory, self.values, rewards):
            # a.reinforce(v.data.view(1)[0] - loss)
            total_loss += (v.npvalue()[0] - loss) * p_a

            # TODO: loss should live in the VFA, similar to policy
            #total_loss += F.smooth_l1_loss(v, torch.autograd.Variable(torch.Tensor([r])))
            total_loss += dy.squared_distance(v, dy.inputTensor([r]))

        total_loss.forward()
        total_loss.backward()
        #torch.autograd.backward([value_loss] + self.trajectory,
        #                        [torch.ones(1)] + [None]*len(self.trajectory))

#    @profile
    def __call__(self, state):
        action, p_action = self.policy.stochastic_with_probability(state)
        value = self.baseline(state)

        # log actions and values taken along current trajectory
        self.trajectory.append((action, p_action))
        self.values.append(value)

        return action
        #return action.data.view(1)[0]   # return an integer
