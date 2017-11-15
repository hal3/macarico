# https://esc.fnwi.uva.nl/thesis/centraal/files/f355157274.pdf
# http://www.jmlr.org/papers/volume6/wingate05a/wingate05a.pdf

from __future__ import division

import random
import numpy as np
import dynet as dy
import macarico
from enum import Enum
import macarico

help = 1
network_size = 3

# modulo operator (ring topology)
class Network(object):
    def __init__(self):
        self.default_computer_status = np.array([0]*network_size) # 3 computers status

        # sysadmin variables
        self.small_prob_failure = .075     # prob of any computer changing from working to failing
        self.incr_failing_neighbor = 0.125 # failure increase due to connected to failing computer
        self.gamma = 0.95
        self.n_actions = network_size + 1 

        # Probability of failing each round
        self.failing_default = [1-self.small_prob_failure, self.small_prob_failure]
        self.prob_failure = np.array([self.failing_default,]*network_size)

    def mk_env(self):
        self.default_computer_status = np.array([0] *network_size) # 3 computers status
        self.prob_failure = np.array([self.failing_default,]*network_size)
        return SysAdmin(self)

class SysAdmin(macarico.Env):
    
    def __init__(self, network):
        self.network = network
        self.t = 0
        self.reward = 0
        self.reward_array = []
        self.discount = 1
        self.comp_status = network.default_computer_status[:]
        self.random_seeds = None

        # For macarico.Env
        self.T = 20 # Horizon
        self.n_actions = network_size + 1
        self.actions = range(self.n_actions)

    def run_episode(self, policy):
        self.random_seeds = np.array([np.random.RandomState(0), np.random.RandomState(10), np.random.RandomState(8)])
        self.output = []
        for self.t in range(self.T):
            if help:
                print "\nt: ", str(self.t) , " --> ", np.array_str(self.comp_status)
            a = policy(self)
            # During each step the agent can do nothing or reboot any of the computers
            a, r = self.step(a)
            self.output.append(a)
            self.reward += self.discount * np.sum(1  * (1-self.comp_status))
            self.reward += self.discount * np.sum(-2 * self.comp_status)

            this_reward = self.discount * np.sum(-2 * self.comp_status) + \
                          self.discount * np.sum(1-self.comp_status) + \
                          r
                         
            self.reward_array.append(this_reward) 
            if help:
                print "\t\t Reward --> ", str(self.reward)
        if help:
            print (" ------------------\n")
            print "\t state  --> ", np.array_str(self.comp_status)
            print "\t Reward --> ", str(self.reward)
            print "Done! ----> episode"
            print "\t --------------------"
        return self.output, self.reward

    def step(self, action):
        tmp_reward = 0
        # computer can start to fail with a small chance
        # probability of computer failing randomly .075
        fail_chance = []
        for idx, (prob_succ, prob_fail) in enumerate(self.network.prob_failure):
            fail_chance.append(self.random_seeds[idx].choice([0,1], p=[prob_succ, prob_fail], size=(1))[0])

        fail_chance = fail_chance | self.comp_status

        # If a computer is connected to a failing com
        for idx, val in enumerate(fail_chance):
            if val:
                if help:
                    print "\t fail: [", str(idx), "]"
                self.comp_status[idx] = 1
                for nbr in [(idx+1) % 3, (idx-1) % 3]:
                    if help:
                        print "\t\t Neighbor Failure Increase: [", str(nbr), "]"
                    self.network.prob_failure[nbr][0] -= self.network.incr_failing_neighbor
                    self.network.prob_failure[nbr][1] += self.network.incr_failing_neighbor    
                    if help:
                        print "\t\t network.prob_failure: ", str(self.network.prob_failure[nbr])
                    
                    self.network.prob_failure[nbr][0] = max([self.network.prob_failure[nbr][0], 0])
                    self.network.prob_failure[nbr][1] = min([self.network.prob_failure[nbr][1], 1])

        #Last action is to do nothing
        #Else reboot the computer choosen
        if action != (self.n_actions-1):
            if help:
                print "\t fix: [", str(action), "]"
            self.comp_status[action] = 0
            self.network.prob_failure[action] = self.network.failing_default
            if help:
                print "\t\t network.prob_failure: ", str(self.network.prob_failure[action])
            self.reward += self.discount * -2.0 
            tmp_reward = self.discount * -2.0
            
        else:
            if help:
                print "\t fix: None - ", str(action)

        return action, tmp_reward

class SysAdminLoss(macarico.Loss):
    def __init__(self):
        super(SysAdminLoss, self).__init__('reward')

    def evaluate(self, ex, state):
        return (-1) * np.array(state.reward_array)

class SysAdminFeatures(macarico.Features):
    def __init__(self):
       macarico.Features.__init__(self, 'computers', network_size)

    def forward(self, state):
        view = np.reshape(state.comp_status, (1,network_size))
        return dy.inputTensor(view)
