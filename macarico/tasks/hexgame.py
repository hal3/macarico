from __future__ import division
import random
import macarico

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var

class Hex(macarico.Env):
    """
    largely based on the openai gym implementation
    """
    BLACK, WHITE = 0, 1
    
    def __init__(self, player_color, board_size):
        self.board_size = board_size
        self.player_color = player_color
        self.n_actions = self.board_size ** 2 + 1
        self.actions = range(self.n_actions)
        self.state = torch.zeros((3, self.board_size, self.board_size))
        self.to_play = Hex.BLACK
        self.T = 100
        
    def mk_env(self):
        self.state *= 0
        self.state[2,:,:] = 1.0
        self.to_play = Hex.BLACK
        return self

    def run_episode(self, policy):
        self.output = []

        if self.player_color != self.to_play:
            self.actions = get_possible_actions(self.state)
            a = np.random.choice(self.actions)
            make_move(self.state, a, Hex.BLACK)
            self.to_play = Hex.WHITE

        self.reward = 0
            
        for self.t in xrange(self.T):
            self.actions = get_possible_actions(self.state)
            a = policy(self)
            self.output.append(a)
            if resign_move(self.board_size, a):
                self.reward = -1
                break
            if not valid_move(self.state, a):
                self.reward = -1
                break
            make_move(self.state, a, self.player_color)

            self.actions = get_possible_actions(self.state)
            a = None if len(self.actions) == 0 else np.random.choice(self.actions) 
            if a is not None:
                if resign_move(self.board_size, a):
                    self.reward = 1
                    break
                make_move(self.state, a, 1 - self.player_color)

            self.reward = game_finished(self.state)
            if self.reward != 0:
                if self.player_color == Hex.WHITE:
                    self.reward = - self.reward
                break

        return self.output


def resign_move(board_size, a): return a == board_size**2
def valid_move(board, a):
    coords = action_to_coord(board, a)
    return board[2, coords[0], coords[1]] == 1
def make_move(board, a, player):
    coords = action_to_coord(board, a)
    board[2, coords[0], coords[1]] = 0
    board[player, coords[0], coords[1]] = 1
def coord_to_action(board, coords):
    return coords[0] * board.shape[-1] + coords[1]
def action_to_coord(board, a):
    return a // board.shape[-1], a % board.shape[-1]
def get_possible_actions(board):
    free_x, free_y = np.where(board[2,:,:].numpy() == 1)
    return [coord_to_action(board, [x,y]) for x, y in zip(free_x, free_y)]
def game_finished(board):
    d = board.shape[1]
    inpath = set()
    newset = set()
    for i in range(d):
        if board[0, 0, i] == 1:
            newset.add(i)

    while len(newset) > 0:
        for i in range(len(newset)):
            v = newset.pop()
            inpath.add(v)
            cx = v // d
            cy = v % d
            # Left
            if cy > 0 and board[0, cx, cy - 1] == 1:
                v = cx * d + cy - 1
                if v not in inpath:
                    newset.add(v)
            # Right
            if cy + 1 < d and board[0, cx, cy + 1] == 1:
                v = cx * d + cy + 1
                if v not in inpath:
                    newset.add(v)
            # Up
            if cx > 0 and board[0, cx - 1, cy] == 1:
                v = (cx - 1) * d + cy
                if v not in inpath:
                    newset.add(v)
            # Down
            if cx + 1 < d and board[0, cx + 1, cy] == 1:
                if cx + 1 == d - 1:
                    return 1
                v = (cx + 1) * d + cy
                if v not in inpath:
                    newset.add(v)
            # Up Right
            if cx > 0 and cy + 1 < d and board[0, cx - 1, cy + 1] == 1:
                v = (cx - 1) * d + cy + 1
                if v not in inpath:
                    newset.add(v)
            # Down Left
            if cx + 1 < d and cy > 0 and board[0, cx + 1, cy - 1] == 1:
                if cx + 1 == d - 1:
                    return 1
                v = (cx + 1) * d + cy - 1
                if v not in inpath:
                    newset.add(v)

    inpath.clear()
    newset.clear()
    for i in range(d):
        if board[1, i, 0] == 1:
            newset.add(i)

    while len(newset) > 0:
        for i in range(len(newset)):
            v = newset.pop()
            inpath.add(v)
            cy = v // d
            cx = v % d
            # Left
            if cy > 0 and board[1, cx, cy - 1] == 1:
                v = (cy - 1) * d + cx
                if v not in inpath:
                    newset.add(v)
            # Right
            if cy + 1 < d and board[1, cx, cy + 1] == 1:
                if cy + 1 == d - 1:
                    return -1
                v = (cy + 1) * d + cx
                if v not in inpath:
                    newset.add(v)
            # Up
            if cx > 0 and board[1, cx - 1, cy] == 1:
                v = cy * d + cx - 1
                if v not in inpath:
                    newset.add(v)
            # Down
            if cx + 1 < d and board[1, cx + 1, cy] == 1:
                v = cy * d + cx + 1
                if v not in inpath:
                    newset.add(v)
            # Up Right
            if cx > 0 and cy + 1 < d and board[1, cx - 1, cy + 1] == 1:
                if cy + 1 == d - 1:
                    return -1
                v = (cy + 1) * d + cx - 1
                if v not in inpath:
                    newset.add(v)
            # Left Down
            if cx + 1 < d and cy > 0 and board[1, cx + 1, cy - 1] == 1:
                v = (cy - 1) * d + cx + 1
                if v not in inpath:
                    newset.add(v)
    return 0    

    
class HexLoss(macarico.Loss):
    def __init__(self):
        super(HexLoss, self).__init__('-reward')
    def evaluate(self, ex, state):
        return -state.reward

class HexFeatures(macarico.Features):
    def __init__(self, board_size):
        self.board_size = board_size
        macarico.Features.__init__(self, 'hex', 3 * self.board_size ** 2)

    def forward(self, state):
        view = state.state.view(1, 1, 3 * self.board_size ** 2)
        return Var(view)
