from __future__ import division
import random
import macarico
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var

class Pendulum(macarico.Env):
    """
    largely based on the openai gym implementation:
      https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
    """
    def __init__(self):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = 0.01
        self.granularity = 0.1 # controls how many actions there are
        self.action_torques = np.arange(-2, 2+self.granularity, self.granularity)
        self.n_actions = len(self.action_torques)
        self.actions = range(self.n_actions)
        self.th, self.th_dot = 0., 0.
        self.T = 100
        self.loss = 0.
        
    def mk_env(self):
        self.th = np.random.uniform(-np.pi, np.pi)
        self.th_dot = np.random.uniform(-1, 1)
        self.loss = 0.
        return self

    def run_episode(self, policy):
        self.output = []
        for self.t in xrange(self.T):
            a = policy(self)
            self.output.append((a))#, self.th))
            u = self.action_torques[a] # + np.random.uniform(low=-self.granularity/2, high=self.granularity/2)
            self.step(u)
#        print
        return self.output

    def step(self, u):
        g, m, l = 10., 1., 1.
        cost = angle_normalize(self.th) ** 2 + .1 * self.th_dot ** 2 + 0.001 * (u**2)
#        print cost, angle_normalize(self.th), self.th_dot, u, torch.sin(self.th + torch.pi)
        self.loss += cost

        new_th_dot = self.th_dot + self.dt * \
                     (-3*g/(2*l) * np.sin(self.th + np.pi) + 3./(m*l**2)*u)
        self.th = self.th + new_th_dot * self.dt
        self.th_dot = np.clip(new_th_dot, -self.max_speed, self.max_speed)

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)


class PendulumLoss(macarico.Loss):
    def __init__(self):
        super(PendulumLoss, self).__init__('cost/T')
    def evaluate(self, ex, state):
        return state.loss / state.T

class PendulumFeatures(macarico.Features):
    def __init__(self):
        macarico.Features.__init__(self, 'pendulum', 4)

    def forward(self, state):
        view = torch.zeros((1, 1, 4))
        view[0,0,0] = 1.
        view[0,0,1] = np.cos(state.th)
        view[0,0,2] = np.sin(state.th)
        view[0,0,3] = state.th_dot
        return Var(view)
