# -*- coding: utf-8 -*-
from __future__ import division, generators, print_function
import random
import macarico

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import macarico.util as util
from macarico.util import Var, Varng

NORTH, SOUTH, EAST, WEST = 0, 1, 2, 3
PASSABLE, SEED, POWER = 0, 1, 2

def str_direction(d):
    if   d == NORTH: return '^'
    elif d == SOUTH: return 'v'
    elif d == EAST:  return '>'
    elif d == WEST:  return '<'

def coord_dir(A, d):
    x, y = A
    if   d == NORTH: return int(x), int(y-1)
    elif d == SOUTH: return int(x), int(y+1)
    elif d == EAST:  return int(x+1), int(y)
    elif d == WEST:  return int(x-1), int(y)
    return None

def set_flag(obs, f):
    return obs | (1 << f)

def manhattan_distance(A, B):
    return abs(A[0] - B[0]) + abs(A[1] - B[1])

def directional_distance(A, B, d):
    if d == NORTH: return B[1] - A[1]
    if d == SOUTH: return A[1] - B[1]
    if d == EAST:  return B[0] - A[0]
    if d == WEST:  return A[0] - B[0]

def opposite(d):
    if d == NORTH: return SOUTH
    if d == SOUTH: return NORTH
    if d == EAST:  return WEST
    if d == WEST:  return EAST
    return -1

import sys, tty, termios
def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch
    
def keyboard_policy(state):
    print()
    print(state)
    while True:
        c = getch()
        if c in 'wi': return NORTH
        if c in 'aj': return WEST
        if c in 'sk': return SOUTH
        if c in 'dl': return EAST
        if c in 'q': return None
    
class POCMAN(macarico.Env):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.smell_range = 1
        self.hear_range = 2
        self.food_prob = 0.5
        self.chase_prob = 0.75
        self.defensive_slip = 0.25
        self.power_num_steps = 15
        self.reward_clear_level = 1000 / 1000
        self.reward_default = -1 / 1000
        self.reward_eat_food = 10 / 1000
        self.reward_eat_ghost = 25 / 1000
        self.reward_hit_wall = -25 / 1000
        self.reward_die = -100 / 1000
        self.n_actions = 4
        self.n_observations = 1 << 10
        self.gamma = 0.95
        self.maze = None
        self.passage_y = -1
        self.ghost_home = (0, 0)
        self.pocman_home = (0, 0)
        self.ghost_pos = []
        self.ghost_dir = []
        self.food = set()
        self.num_ghosts = 0
        self.num_food = 0
        self.power_steps = 0
        self.pocman = (0, 0)
        self.ghost_range = 1
        self.actions = set([0,1,2,3])
        self.T = 100000
        self.t = 0
        self.total_reward = 0
        self._trajectory = []
        self.obs = [0]

    def horizon(self): return self.T
        
    def set_maze(self, maze_strings):
        self.maze = [list(map(int, s)) for s in maze_strings]

    def __str__(self):
        ghost_positions = set(self.ghost_pos)
        s = ''
        s += '⬛' * (self.width + 2) + ' R=' + str(self.total_reward) + '\n'
        for y in range(self.height):
            if y == self.passage_y:
                s += '<'
            else:
                s += '⬛'
            for x in range(self.width):
                c = ' '
                if not self.passable((x, y)):
                    c = '⬛'
                if (x,y) in self.food:
                    c = '⬤' if self.check((x, y), POWER) else '·'
                if (x,y) in ghost_positions:
                    c = 'X' if self.pocman == (x,y) else \
                        'ᗣ' if self.power_steps == 0 else \
                        'ᗢ'
                elif (x,y) == self.pocman:
                    c = 'ᗦ' if self.power_steps > 0 else 'ᗤ'
                s += c
            if y == self.passage_y:
                s += '>'
            else:
                s += '⬛'
            if y == 0: s += ' t=' + str(self.t)
            if y == 1 and len(self._trajectory) > 0: s += ' a=' + str_direction(self._trajectory[-1])
            if y == 2 and self.obs[-1] & (1 << 0) : s += ' g' + str_direction(0)
            if y == 3 and self.obs[-1] & (1 << 1) : s += ' g' + str_direction(1)
            if y == 4 and self.obs[-1] & (1 << 2) : s += ' g' + str_direction(2)
            if y == 5 and self.obs[-1] & (1 << 3) : s += ' g' + str_direction(3)
            if y == 6 and self.obs[-1] & (1 << 4) : s += ' p' + str_direction(0)
            if y == 7 and self.obs[-1] & (1 << 5) : s += ' p' + str_direction(1)
            if y == 8 and self.obs[-1] & (1 << 6) : s += ' p' + str_direction(2)
            if y == 9 and self.obs[-1] & (1 << 7) : s += ' p' + str_direction(3)
            if y ==10 and self.obs[-1] & (1 << 8) : s += ' food'
            if y ==11 and self.obs[-1] & (1 << 9) : s += ' ghost'
            s += '\n'
        s += '⬛' * (self.width + 2) + '\n'
        return s
        
    def mk_env(self):
        self.pocman = self.pocman_home
        self.ghost_pos = []
        for g in range(self.num_ghosts):
            x, y = self.ghost_home
            x += g % 2
            y += g / 2
            self.ghost_pos.append((x, y))
        self.ghost_dir = [-1] * self.num_ghosts
        self.food = set()
        for x in range(self.width):
            for y in range(self.height):
                if self.check((x, y), SEED) and \
                   (self.check((x, y), POWER) or \
                    random.random() < self.food_prob):
                    self.food.add((x, y))
        self.num_food = len(self.food)
        self.power_steps = 0
        self.t = 0
        self.total_reward = 0
        self._trajectory = []
        return self

    def _run_episode(self, policy, print_it=False):
        self._trajectory = []
        self.obs = [self.make_observations()]
        discount = 1
        for self.t in range(self.T):
            if print_it:
                print(self)
                time.sleep(0.1)
            a = policy(self)
            if a is None:
                break
            self._trajectory.append(a)
            done, reward = self.step(a)
            self.total_reward += reward * discount
            if done:
                break
            discount *= self.gamma
        if self.t < 2:
            print('eek')
        return str(self.t) + ': ' +  ''.join(map(str_direction, self._trajectory))

    def check(self, pos, item):
        x, y = pos
        return (self.maze[y][x] & (1 << item)) != 0

    def inside(self, pos):
        x, y = pos
        return 0 <= x and x < self.width and 0 <= y and y < self.height

    def passable(self, pos):
        return self.check(pos, PASSABLE)

    def next_pos(self, A, d):
        x, y = A
        B = None
        if x == 0 and y == self.passage_y and d == WEST:
            B = self.width-1, y
        elif x == self.width - 1 and y == self.passage_y and d == EAST:
            B = 0, y
        else:
            B = coord_dir(A, d)
            
        if self.inside(B) and self.passable(B):
            return B
        return None

    def step(self, a):
        reward = self.reward_default
        new_pos = self.next_pos(self.pocman, a)
        if new_pos is None:
            reward += self.reward_hit_wall
        else:
            self.pocman = new_pos

        if self.power_steps > 0:
            self.power_steps -= 1

        hit_ghost = set()
        for g in range(self.num_ghosts):
            if self.pocman == self.ghost_pos[g]:
                hit_ghost.add(g)
            self.move_ghost(g)
            if self.pocman == self.ghost_pos[g]:
                hit_ghost.add(g)
                
        if len(hit_ghost) > 0:
            if self.power_steps > 0:
                for g in hit_ghost:
                    reward += self.reward_eat_ghost
                    self.ghost_pos[g] = self.ghost_home
                    self.ghost_dir[g] = -1
            else:
                reward += self.reward_die
                return True, reward

        self.obs.append(self.make_observations())
        
        if self.pocman in self.food:
            self.food.remove(self.pocman)
            self.num_food -= 1
            if self.num_food == 0:
                reward += self.reward_clear_level
                return True, reward
            if self.check(self.pocman, POWER):
                self.power_steps = self.power_num_steps
            reward += self.reward_eat_food

        return False, reward

    def make_observations(self):
        obs = 0
        for d in range(4):
            if self.see_ghost(d) >= 0:
                obs = set_flag(obs, d)
            wpos = self.next_pos(self.pocman, d)
            if wpos is not None and self.passable(wpos):
                obs = set_flag(obs, d+4)
        if self.smell_food():
            obs = set_flag(obs, 8)
        if self.hear_ghost():
            obs = set_flag(obs, 9)
        return obs

    def move_ghost(self, g):
        if manhattan_distance(self.pocman, self.ghost_pos[g]) < self.ghost_range:
            if self.power_steps > 0:
                self.move_ghost_defensive(g)
            else:
                self.move_ghost_aggressive(g)
        else:
            self.move_ghost_random(g)

    def move_ghost_aggressive(self, g):
        if random.random() > self.chase_prob:
            self.move_ghost_random(g)
            return

        best_dist = self.width + self.height
        best_pos = self.ghost_pos[g]
        best_dir = -1
        for d in range(4):
            dist = directional_distance(self.pocman, self.ghost_pos[g], d)
            newpos = self.next_pos(self.ghost_pos[g], d)
            if dist <= best_dist and newpos is not None and opposite(d) != self.ghost_dir[g]:
                best_dist = dist
                best_pos = newpos
                best_dir = d
                # TODO: updating best_dir wasn't in pocman.cpp???
        self.ghost_pos[g] = best_pos
        self.ghost_dir[g] = best_dir

    def move_ghost_defensive(self, g):
        if random.random() < self.defensive_slip and self.ghost_dir[g] >= 0:
            self.ghost_dir[g] = -1
            return

        best_dist = 0
        best_pos = self.ghost_pos[g]
        best_dir = -1
        for d in range(4):
            dist = directional_distance(self.pocman, self.ghost_pos[g], d)
            newpos = self.next_pos(self.ghost_pos[g], d)
            if dist >= best_dist and newpos is not None and opposite(d) != self.ghost_dir[g]:
                best_dist = dist
                best_pos = newpos
                best_dir = d
        self.ghost_pos[g] = best_pos
        self.ghost_dir[g] = best_dir

    def move_ghost_random(self, g):
        newpos = None
        d = 0
        num_trials = 0
        while True:
            d = int(random.random() * 4)
            newpos = self.next_pos(self.ghost_pos[g], d)
            if d != opposite(self.ghost_dir[g]) and newpos is not None:
                break
            num_trials += 1
            if num_trials > 200:
                print('too many trials')
                from arsenal import ip; ip()
        self.ghost_pos[g] = newpos
        self.ghost_dir[g] = d

    def see_ghost(self, d):
        eyepos = coord_dir(self.pocman, d)
        while self.inside(eyepos) and self.passable(eyepos):
            for g, pos in enumerate(self.ghost_pos):
                if pos == eyepos:
                    return g
            eyepos = coord_dir(eyepos, d)
        return -1

    def hear_ghost(self):
        for pos in self.ghost_pos:
            if manhattan_distance(pos, self.pocman) <= self.hear_range:
                return True
        return False

    def smell_food(self):
        x, y = self.pocman
        for xd in range(-self.smell_range, self.smell_range+1):
            for yd in range(-self.smell_range, self.smell_range+1):
                pos = x+xd, y+yd
                if self.inside(pos) and pos in self.food:
                    return True
        return False    
        
class MicroPOCMAN(POCMAN):
    def __init__(self):
        super(MicroPOCMAN, self).__init__(7, 7)
        self.num_ghosts = 1
        self.ghost_range = 3
        self.pocman_home = (3, 0)
        self.ghost_home = (3, 4)
        self.passage_y = 5
        self.set_maze(['3333333',
                       '3303033',
                       '3033303',
                       '3330333',
                       '3033303',
                       '3303033',
                       '3333333'])

class MiniPOCMAN(POCMAN):
    def __init__(self):
        super(MiniPOCMAN, self).__init__(10, 10)
        self.num_ghosts = 3
        self.ghost_range = 4
        self.pocman_home = (4, 2)
        self.ghost_home = (4, 4)
        self.passage_y = 5
        self.set_maze(['3333333333',
                       '3003003003',
                       '3033333303',
                       '3330000333',
                       '0030113300',
                       '0030113300',
                       '3330000333',
                       '3033333303',
                       '3003003003',
                       '3333333333'])

class FullPOCMAN(POCMAN):
    def __init__(self):
        super(FullPOCMAN, self).__init__(17, 19)
        self.num_ghosts = 4
        self.ghost_range = 6
        self.pocman_home = (8, 6)
        self.ghost_home = (8, 10)
        self.passage_y = 8
        self.set_maze(['33333333333333333',
                       '30030003030003003',
                       '73333333333333337',
                       '30030300000303003',
                       '33330333033303333',
                       '00030003030003000',
                       '00030111111103000',
                       '00030101110103000',
                       '11130101110103111',
                       '00030100000103000',
                       '00030111111103000',
                       '00030300000303000',
                       '33333333333333333',
                       '30030003030003003',
                       '73033333333333037',
                       '03030300000303030',
                       '33330333033303333',
                       '30000003030000003',
                       '33333333333333333'])

class POCLoss(macarico.Loss):
    def __init__(self):
        super(POCLoss, self).__init__('-reward')
        
    def evaluate(self, ex, state):
        return -state.total_reward

class LocalPOCFeatures(macarico.StaticFeatures):
    def __init__(self, history_length=1):
        macarico.StaticFeatures.__init__(self, 10*history_length)
        self.history_length = history_length
        self._t = nn.Linear(1,1,bias=False)

    def _forward(self, state):
        view = util.zeros(self._t, 1, 1, 10 * self.history_length)
        for h in range(self.history_length):
            obs = state.obs[max(0, len(state.obs)-h-1)]
            for i in range(10):
                if (obs & i) > 0:
                    view[0, 0, h * 10 + i] = 1.
        return Varng(view)
    

class GlobalPOCFeatures(macarico.StaticFeatures):
    # Empty:0, Block:1, Pellet:2, Food:3, Pac:4, Ghost:5, PowerPac:6, ScaredGhost:7
    def __init__(self, width, height):
        macarico.StaticFeatures.__init__(self, width*height*8)
        self.width = width
        self.height = height
        self._t = nn.Linear(1,1,bias=False)

    def _forward(self, state):
        ghost_positions = set(state.ghost_pos)
        view = util.zeros(self._t, 1, 1, 10 * self.history_length)
        for y in range(self.height):
            for x in range(self.width):
                idx = (x * self.height + y) * 8
                c = 0
                pos = (x, y)
                if not state.passable(pos): c = 1
                if pos in state.food:
                    c = 3 if state.check(pos, POWER) else 2
                if pos in ghost_positions:
                    c = 5 if state.power_steps == 0 else 7
                elif pos == state.pocman:
                    c = 4 if state.power_steps == 0 else 6
                view[0, 0, idx + c] = 1
        return Varng(view)

class POCReference(macarico.Reference):
    def __call__(self, state):
        good = set()
        obs = state.obs[-1]
        last_action = -1 if len(state._trajectory) == 0 else state._trajectory[-1]
        if state.power_steps > 0 and (obs & 15) != 0:
            # power pill and can see a ghost => chase it
            for a in range(4):
                if (obs & (1 << a)) != 0:
                    good.add(a)
        else:
            # otherwise, avoid ghosts and avoid changing direction
            for a in range(4):
                newp = state.next_pos(state.pocman, a)
                if newp and (obs & (1 << a)) == 0 and opposite(a) != last_action:
                    good.add(a)
        if len(good) > 0:
            return random.choice(list(good))
        else:
            return int(random.random() * 4)
    
def play_game():
    pocman = FullPOCMAN()
    env = pocman.mk_env()
    env.run_episode(keyboard_policy)
    
