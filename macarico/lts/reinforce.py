from __future__ import division

import random
#import torch
#import torch.nn.functional as F
#from torch import autograd
#from torch.autograd import Variable

import macarico


class Reinforce(macarico.Learner):
    "REINFORCE with a scalar baseline function."

    def __init__(self, policy, baseline, max_deviations=None, uniform=False):
        self.trajectory = []
        self.baseline = baseline
        self.policy = policy
        self.max_deviations = max_deviations
        self.uniform = uniform
        self.t = None
        self.dev_t = None
        super(Reinforce, self).__init__()

    def update(self, loss):
        b = self.baseline()
        for a in self.trajectory:
            a.reinforce(b - loss)
        self.baseline.update(loss)
        torch.autograd.backward(self.trajectory[:], [None]*len(self.trajectory))

    def __call__(self, state):
        if self.t is None:
            self.t = 0
            if self.max_deviations is not None:
                t_list = range(1, state.T+1)
                random.shuffle(t_list)
                self.dev_t = set(t_list[:self.max_deviations])

        self.t += 1
        if self.max_deviations is None or self.t in self.dev_t:
            action = self.policy.stochastic(state, 1000 if self.uniform else 1)
            # log actions (and values for actor critic) taken along current trajectory
            self.trajectory.append(action)
            return action.data.view(1)[0] # return an integer
        else:
            action = self.policy.greedy(state)
            return action


# TODO: scalar baseline should be an instance of this class with one constant feature.
class LinearValueFn(torch.nn.Module):
    """
    Linear value function regressor.
    """

    def __init__(self, features):
        torch.nn.Module.__init__(self)
        self._predict = torch.nn.Linear(features.dim, 1)
        self.features = features

#    @profile
    def __call__(self, state):
        return self._predict(self.features(state))


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

        value_loss = 0.0
        for a, v, r in zip(self.trajectory, self.values, rewards):
            a.reinforce(v.data.view(1)[0] - loss)

            # TODO: loss should live in the VFA, similar to policy
            value_loss += F.smooth_l1_loss(v, torch.autograd.Variable(torch.Tensor([r])))

#        value_loss = 0   # for training value function regression
#        for a, v, r in zip(self.saved_actions, self.saved_values, rewards):
#            [vv] = v.data.squeeze()

        torch.autograd.backward([value_loss] + self.trajectory,
                                [torch.ones(1)] + [None]*len(self.trajectory))

#    @profile
    def __call__(self, state):
        action = self.policy.stochastic(state)
        value = self.baseline(state)

        # log actions and values taken along current trajectory
        self.trajectory.append(action)
        self.values.append(value)

        return action.data.view(1)[0]   # return an integer
