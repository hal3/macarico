from __future__ import division, generators, print_function
import random
import macarico
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import macarico.util as util
from macarico.util import Var, Varng

class Pendulum(macarico.Env):
    """
    largely based on the openai gym implementation:
      https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
    """
    def __init__(self):
        self.granularity = 0.1 # controls how many actions there are
        self.action_torques = np.arange(-2, 2+self.granularity, self.granularity)
        macarico.Env.__init__(self, len(self.action_torques), 100)
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = 0.01
        self.actions = range(self.n_actions)
        self.th, self.th_dot = 0., 0.
        self.example.loss = 0.
        self.example.T = self.T
        
    def _run_episode(self, policy):
        self.th = np.random.uniform(-np.pi, np.pi)
        self.th_dot = np.random.uniform(-1, 1)
        self.example.loss = 0.
        for _ in range(self.horizon()):
            a = policy(self)
            u = self.action_torques[a] # + np.random.uniform(low=-self.granularity/2, high=self.granularity/2)
            self.step(u)
        return self._trajectory

    def step(self, u):
        g, m, l = 10., 1., 1.
        cost = angle_normalize(self.th) ** 2 + .1 * self.th_dot ** 2 + 0.001 * (u**2)
#        print cost, angle_normalize(self.th), self.th_dot, u, torch.sin(self.th + torch.pi)
        self.example.loss += cost

        new_th_dot = self.th_dot + self.dt * \
                     (-3*g/(2*l) * np.sin(self.th + np.pi) + 3./(m*l**2)*u)
        self.th = self.th + new_th_dot * self.dt
        self.th_dot = np.clip(new_th_dot, -self.max_speed, self.max_speed)

    def _rewind(self):
        pass

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)


class PendulumLoss(macarico.Loss):
    def __init__(self):
        super(PendulumLoss, self).__init__('cost/T')
    def evaluate(self, example):
        return example.loss / example.T

class PendulumFeatures(macarico.StaticFeatures):
    def __init__(self):
        macarico.StaticFeatures.__init__(self, 4)
        self._t = nn.Linear(1,1,bias=False)

    def _forward(self, state):
        view = util.zeros(self._t.weight, 1, 1, 4)
        view[0,0,0] = 1.
        view[0,0,1] = np.cos(state.th)
        view[0,0,2] = np.sin(state.th)
        view[0,0,3] = state.th_dot
        return Varng(view)
