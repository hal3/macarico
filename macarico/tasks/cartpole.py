"""
Largely based on the OpenAI Gym Implementation
https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
"""
from __future__ import division

import numpy as np
import macarico
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var

class CartPoleEnv(macarico.Env):
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds
        self.state = None
        # For macarico.Env
        self.T = 200
        self.n_actions = 2
        self.actions = range(self.n_actions)

    def mk_env(self):
        self.reset()
        return self

    def reset(self):
        self.state = torch.rand(4) * 0.1 - 0.05
        self.steps_beyond_done = None
        return self.state

    def run_episode(self, policy):
        self.output = []
        for self.t in range(self.T):
            a = policy(self)
            self.output.append((a))
            if self.step(a):
                break
        return self.output

    def step(self, action):
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action == 1 else - self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        #self.state = (x, x_dot, theta, theta_dot)
        self.state[0] = x
        self.state[1] = x_dot
        self.state[2] = theta
        self.state[3] = theta_dot
        done = x < -self.x_threshold \
            or x > self.x_threshold \
            or theta < -self.theta_threshold_radians \
            or theta > self.theta_threshold_radians
        return done


class CartPoleLoss(macarico.Loss):
    def __init__(self):
        super(CartPoleLoss, self).__init__('-t')

    def evaluate(self, ex, state):
        return -state.t
        #return (100 - state.t) / 100


class CartPoleFeatures(macarico.Features):
    def __init__(self):
        macarico.Features.__init__(self, 'cartpole', 4)

    def forward(self, state):
        #view = torch.reshape(state.state, (1, 4))
        #return dy.inputTensor(view)
        return Var(state.state.view(1,1,-1), requires_grad=False)
