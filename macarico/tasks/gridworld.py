from __future__ import division

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
import macarico

class Example(object):
    def __init__(self, width, height, start, walls, terminals, per_step_cost, max_steps, gamma, p_step_success):
        self.width = width
        self.height = height
        self.start = start
        self.walls = walls
        self.terminal = terminals
        self.per_step_cost = per_step_cost
        self.max_steps = max_steps
        self.gamma = gamma
        self.p_step_success = p_step_success

    def mk_env(self):
        return GridWorld(self)

def make_default_gridworld(per_step_cost=0.05, max_steps=50, gamma=0.99, p_step_success=0.8, start_random=False):
    #    0123
    #   0   +
    #   1 # -
    #   2 #
    #   3
    start = (0, 3)
    if start_random:
        start = (random.randint(0,3), random.randint(0,3))
    return Example(4, 4, start, set([(1,1),(1,2)]), {(3,0): 1, (3,1): -1},
                   per_step_cost, max_steps, gamma, p_step_success)
    
def make_big_gridworld(per_step_cost=0.01, max_steps=200, gamma=0.99, p_step_success=0.9):
    # from http://cs.stanford.edu/people/karpathy/reinforcejs/
    return Example(10, 10, (0,9),
                   set([(1,2), (2,2), (3,2), (4,2), (6,2), (7,2), (8,2),
                        (4,3), (4,4), (4,5), (4,6), (4,7)]),
                   {(3,3): -1, (3,7): -1, (5,4): -1, (5,5): 1, (6,5): -1, (6,6): -1, 
                    (5,7): -1, (6,7): -1, (8,5): -1, (8,6): -1},
                   per_step_cost, max_steps, gamma, p_step_success)

class GridWorld(macarico.Env):
    UP, LEFT, DOWN, RIGHT = 0, 1, 2, 3
    
    def __init__(self, example):
        self.ex = example
        self.T = example.max_steps
        self.t = 0
        self.loc = example.start
        self.reward = 0.
        self.discount = 1.
        self.output = []
        self.actions = set([self.UP, self.DOWN, self.LEFT, self.RIGHT])
        super(GridWorld, self).__init__(len(self.actions))

    def rewind(self):
        self.t = 0
        self.loc = self.ex.start
        self.reward = 0.
        self.discount = 1.
        
    def run_episode(self, policy):
        self.output = []
        for self.t in xrange(self.T):
            a = policy(self)
            self.output.append(a)
            self.step(a)
            self.reward -= self.discount * self.ex.per_step_cost
            if self.loc in self.ex.terminal:
                self.reward += self.discount * self.ex.terminal[self.loc]
                break
            self.discount *= self.ex.gamma
        return ''.join(map(self.str_direction, self.output))

    def str_direction(self, a):
        return "U" if a == self.UP else \
               "D" if a == self.DOWN else \
               "L" if a == self.LEFT else \
               "R" if a == self.RIGHT else \
               "?"
        
    def step(self, a):
        if random.random() > self.ex.p_step_success:
            # step failure; pick a neighboring action
            a = (a + 2 * ((random.random() < 0.5) - 1)) % 4
        # take the step
        new_loc = list(self.loc)
        if a == self.UP:    new_loc[1] -= 1
        if a == self.DOWN:  new_loc[1] += 1
        if a == self.LEFT:  new_loc[0] -= 1
        if a == self.RIGHT: new_loc[0] += 1
        new_loc = tuple(new_loc)
        if self.is_legal(new_loc):
            self.loc = new_loc
            
    def is_legal(self, new_loc):
        return new_loc[0] >= 0 and new_loc[0] < self.ex.width and \
               new_loc[1] >= 0 and new_loc[1] < self.ex.height and \
               new_loc not in self.ex.walls

class GridLoss(macarico.Loss):
    def __init__(self):
        super(GridLoss, self).__init__('reward')

    def evaluate(self, ex, state):
        return -state.reward
    
class GlobalGridFeatures(macarico.Features):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        macarico.Features.__init__(self, 'grid', width*height)

    def forward(self, state):
        view = torch.zeros((1,self.dim))
        view[0,state.loc[0] * state.ex.height + state.loc[1]] = 1
        return dy.inputTensor(view)

    def __call__(self, state): return self.forward(state)

class LocalGridFeatures(macarico.Features):
    def __init__(self, width, height):
        macarico.Features.__init__(self, 'grid', 4)

    def forward(self, state):
        view = torch.zeros((1,4))
        if not state.is_legal((state.loc[0]-1, state.loc[1]  )): view[0,0] = 1.
        if not state.is_legal((state.loc[0]+1, state.loc[1]  )): view[0,1] = 1.
        if not state.is_legal((state.loc[0]  , state.loc[1]-1)): view[0,2] = 1.
        if not state.is_legal((state.loc[0]  , state.loc[1]+1)): view[0,3] = 1.
        return dy.inputTensor(view)
    
    def __call__(self, state): return self.forward(state)
