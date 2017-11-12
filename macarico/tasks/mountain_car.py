"""
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC

Largely based on the OpenAI Gym Implementation
https://github.com/jaara/ai_examples/blob/master/open_gym/MountainCar-v0.py
"""

import math

import numpy as np
import macarico
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var

class MountainCar(macarico.Env):
    def __init__(self, T=2000):
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.low = np.array([self.min_position, -self.max_speed])
        self.high = np.array([self.max_position, self.max_speed])
        self.reset()
        # TODO what's the correct value for self.T?
        self.T = T
        self.n_actions = 3
        self.actions = range(self.n_actions)

    def mk_env(self):
        self.reset()
        return self

    def run_episode(self, policy):
        self.output = []
        for self.t in range(self.T):
            a = policy(self)
            self.output.append((a))
            if self.step(a):
                break
        return self.output

    def step(self, action):
        position, velocity = self.state
        velocity += (action-1)*0.001 + math.cos(3*position)*(-0.0025)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position == self.min_position and velocity < 0):
            velocity = 0
        done = bool(position >= self.goal_position)
        reward = -1.0

        self.state[0] = position
        self.state[1] = velocity
        return done

    def reset(self):
        self.state = torch.rand(2) - 0.6
        self.state[1] = 0
        #self.state = [np.random.uniform(low=-0.6, high=-0.4), 0]

class MountainCarLoss(macarico.Loss):
    def __init__(self):
        super(MountainCarLoss, self).__init__('t')

    def evaluate(self, ex, state):
        return state.t


class MountainCarFeatures(macarico.Features):
    def __init__(self):
        macarico.Features.__init__(self, 'mountain_car', 2)

    def forward(self, state):
        return Var(state.state.view(1,1,-1))
