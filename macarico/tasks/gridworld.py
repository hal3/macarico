import random

import torch.nn as nn

import macarico
import macarico.util as util
from macarico.util import Varng


class GridSettings(macarico.Example):
    def __init__(self, width, height, start, walls, terminals, per_step_cost, max_steps, gamma, p_step_success):
        super().__init__()
        self.width = width
        self.height = height
        self.start = start
        self.walls = walls
        self.terminal = terminals
        self.per_step_cost = per_step_cost
        self.max_steps = max_steps
        self.gamma = gamma
        self.p_step_success = p_step_success
        self.n_actions = 4


def make_default_gridworld(per_step_cost=0.05, max_steps=50, gamma=0.99, p_step_success=0.8, start_random=False):
    #    0123
    #   0   +
    #   1 # -
    #   2 #
    #   3
    start = (0, 3)
    if start_random:
        start = (random.randint(0, 3), random.randint(0, 3))
    return GridWorld(GridSettings(4, 4, start, {(1, 1),(1, 2)}, {(3, 0): 1, (3, 1): -1},
                                  per_step_cost, max_steps, gamma, p_step_success))


def make_deterministic_gridworld(per_step_cost=0.05, max_steps=50, gamma=1, p_step_success=1, start_random=False):
    #    0123
    #   0   +
    #   1 # -
    #   2 #
    #   3
    start = (0, 3)
    if start_random:
        start = (random.randint(0,3), random.randint(0, 3))
    return GridWorld(GridSettings(4, 4, start, {(1, 1), (1, 2)}, {(3, 0): 1, (3, 1): -1},
                                  per_step_cost, max_steps, gamma, p_step_success))


def make_stochastic_gridworld(per_step_cost=0.0, max_steps=50, gamma=1, p_step_success=0.8, start_random=False):
    #    0123
    #   0   +
    #   1 # -
    #   2 #
    #   3
    start = (0, 3)
    if start_random:
        start = (random.randint(0,3), random.randint(0, 3))
    return GridWorld(GridSettings(4, 4, start, {(1, 1), (1, 2)}, {(3, 0): 1, (3, 1): -1},
                                  per_step_cost, max_steps, gamma, p_step_success))


def make_episodic_gridworld(per_step_cost=0.0, max_steps=50, gamma=1, p_step_success=1, start_random=False):
    #    0123
    #   0   +
    #   1 # -
    #   2 #
    #   3
    start = (0, 3)
    if start_random:
        start = (random.randint(0, 3), random.randint(0, 3))
    return GridWorld(GridSettings(4, 4, start, {(1, 1), (1, 2)}, {(3, 0): 1, (3, 1): -1},
                                  per_step_cost, max_steps, gamma, p_step_success))


# from http://cs.stanford.edu/people/karpathy/reinforcejs/
def make_big_gridworld(per_step_cost=0.01, max_steps=200, gamma=0.99, p_step_success=0.9):
    return GridWorld(GridSettings(
        10, 10, (0, 9), {(1, 2), (2, 2), (3, 2), (4, 2), (6, 2), (7, 2), (8, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7)}
        , {(3, 3): -1, (3, 7): -1, (5, 4): -1, (5, 5): 1, (6, 5): -1, (6, 6): -1, (5, 7): -1, (6, 7): -1, (8, 5): -1,
           (8, 6): -1}, per_step_cost, max_steps, gamma, p_step_success))


class GridWorld(macarico.Env):
    UP, LEFT, DOWN, RIGHT = 0, 1, 2, 3
    
    def __init__(self, example):
        self.loc = example.start
        self.discount = 1.
        self.actions = {self.UP, self.DOWN, self.LEFT, self.RIGHT}
        super(GridWorld, self).__init__(len(self.actions), example.max_steps, example)
        self.example.reward = 0.

    def _rewind(self):
        self.loc = self.example.start
        self.example.reward = 0.
        self.discount = 1.
        
    def _run_episode(self, policy):
        for _ in range(self.horizon()):
            a = policy(self)
            self.step(a)
            self._losses.append(self.discount * self.example.per_step_cost)
            self.example.reward -= self.discount * self.example.per_step_cost
            if self.loc in self.example.terminal:
                self.example.reward += self.discount * self.example.terminal[self.loc]
                self._losses[-1] -= self.discount * self.example.terminal[self.loc]
                break
            self.discount *= self.example.gamma
        return self.output()

    def output(self):
        return ''.join(map(self.str_direction, self._trajectory))

    def str_direction(self, a):
        return "U" if a == self.UP else \
               "D" if a == self.DOWN else \
               "L" if a == self.LEFT else \
               "R" if a == self.RIGHT else \
               "?"
        
    def step(self, a):
        if random.random() > self.example.p_step_success:
            # step failure; pick a neighboring action
            a = (a + 2 * ((random.random() < 0.5) - 1)) % 4
        # take the step
        new_loc = list(self.loc)
        if a == self.UP:
            new_loc[1] -= 1
        if a == self.DOWN:
            new_loc[1] += 1
        if a == self.LEFT:
            new_loc[0] -= 1
        if a == self.RIGHT:
            new_loc[0] += 1
        new_loc = tuple(new_loc)
        if self.is_legal(new_loc):
            self.loc = new_loc
            
    def is_legal(self, new_loc):
        return new_loc[0] >= 0 and new_loc[0] < self.example.width and \
               new_loc[1] >= 0 and new_loc[1] < self.example.height and \
               new_loc not in self.example.walls


class GridLoss(macarico.Loss):
    def __init__(self):
        super(GridLoss, self).__init__('reward')

    def evaluate(self, example):
        return -example.reward


class GlobalGridFeatures(macarico.DynamicFeatures):
    def __init__(self, width, height):
        macarico.DynamicFeatures.__init__(self, width*height)
        self.width = width
        self.height = height
        self._t = nn.Linear(1, 1, bias=False)

    def _forward(self, state):
        view = util.zeros(self._t.weight, 1, 1, self.dim)
        view[0, 0, state.loc[0] * state.example.height + state.loc[1]] = 1
        return Varng(view)

    def __call__(self, state): return self.forward(state)


class LocalGridFeatures(macarico.DynamicFeatures):
    def __init__(self):
        macarico.DynamicFeatures.__init__(self, 4)
        self._t = nn.Linear(1, 1, bias=False)

    def _forward(self, state):
        view = util.zeros(self._t.weight, 1, 1, self.dim)
        if not state.is_legal((state.loc[0]-1, state.loc[1])):
            view[0, 0, 0] = 1.
        if not state.is_legal((state.loc[0]+1, state.loc[1])):
            view[0, 0, 1] = 1.
        if not state.is_legal((state.loc[0], state.loc[1]-1)):
            view[0, 0, 2] = 1.
        if not state.is_legal((state.loc[0], state.loc[1]+1)):
            view[0, 0, 3] = 1.
        return Varng(view)
    
    def __call__(self, state):
        return self.forward(state)
