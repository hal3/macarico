from __future__ import division

import torch
import torch.nn.functional as F
#from torch import autograd
#from torch.autograd import Variable

import macarico


class Reinforce(macarico.LearningAlg):
    "REINFORCE with a scalar baseline function."

    def __init__(self, policy, baseline):
        self.trajectory = []
        self.baseline = baseline
        self.policy = policy
        super(Reinforce, self).__init__()

    def update(self, loss):
        b = self.baseline()
        for a in self.trajectory:
            a.reinforce(b - loss)
        self.baseline.update(loss)
        torch.autograd.backward(self.trajectory[:], [None]*len(self.trajectory))

    def __call__(self, state, limit_actions=None):
        action = self.policy.stochastic(state, limit_actions=limit_actions)
        # log actions (and values for actor critic) taken along current trajectory
        self.trajectory.append(action)
        return action.data[0,0]   # return an integer



# TODO: scalar baseline should be an instance of this class with one constant feature.
class LinearValueFn(torch.nn.Module):
    """
    Linear value function regressor.
    """

    def __init__(self, features):
        torch.nn.Module.__init__(self)
        self._predict = torch.nn.Linear(features.dim, 1)
        self.features = features

    def __call__(self, state):
        return self._predict(self.features(state))


class AdvantageActorCritic(macarico.LearningAlg):

    def __init__(self, policy, state_baseline, gamma=1.0):
        self.policy = policy
        self.baseline = state_baseline
        self.values = []
        self.trajectory = []
        self.gamma = gamma
        super(AdvantageActorCritic, self).__init__()

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

        value_loss = 0.0
        for a, v, r in zip(self.trajectory, self.values, rewards):
            a.reinforce(v.data[0,0] - loss)

            # TODO: loss should live in the VFA, similar to policy
            value_loss += F.smooth_l1_loss(v, torch.autograd.Variable(torch.Tensor([r])))

#        value_loss = 0   # for training value function regression
#        for a, v, r in zip(self.saved_actions, self.saved_values, rewards):
#            [vv] = v.data.squeeze()

        torch.autograd.backward([value_loss] + self.trajectory,
                                [torch.ones(1)] + [None]*len(self.trajectory))

    def __call__(self, state, limit_actions=None):
        action = self.policy.stochastic(state, limit_actions=limit_actions)
        value = self.baseline(state)

        # log actions and values taken along current trajectory
        self.trajectory.append(action)
        self.values.append(value)

        return action.data[0,0]   # return an integer
