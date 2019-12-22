import macarico

import numpy as np
import torch
from macarico.util import Var, Varng

class Hex(macarico.Env):
    """
    largely based on the openai gym implementation
    """
    BLACK, WHITE = 0, 1
    
    def __init__(self, player_color=0, board_size=5):
        macarico.Env.__init__(self, board_size ** 2 + 1, 100)
        self.board_size = board_size
        self.player_color = player_color
        self.actions = range(self.n_actions)
        self.state = torch.zeros((3, self.board_size, self.board_size))
        self.to_play = Hex.BLACK
        self.example.reward = 0
        
    def _rewind(self):
        self.state *= 0
        self.state[2, :, :] = 1.0
        self.to_play = Hex.BLACK
    
    def _run_episode(self, policy):
        if self.player_color != self.to_play:
            self.actions = get_possible_actions(self.state)
            a = np.random.choice(self.actions)
            make_move(self.state, a, Hex.BLACK)
            self.to_play = Hex.WHITE

        self.example.reward = 0
            
        for _ in range(self.horizon()):
            self.actions = get_possible_actions(self.state)
            a = policy(self)
            if resign_move(self.board_size, a):
                self.example.reward = -1
                self._losses.append(1)
                break
            if not valid_move(self.state, a):
                self.example.reward = -1
                self._losses.append(1)
                break
            make_move(self.state, a, self.player_color)

            self.actions = get_possible_actions(self.state)
            a = None if len(self.actions) == 0 else np.random.choice(self.actions) 
            if a is not None:
                if resign_move(self.board_size, a):
                    self.example.reward = 1
                    self._losses.append(-1)
                    break
                make_move(self.state, a, 1 - self.player_color)

            self.example.reward = game_finished(self.state)
            if self.example.reward != 0:
                if self.player_color == Hex.WHITE:
                    self.example.reward = - self.example.reward
                self._losses.append(-self.example.reward)
                break
            else:
                self._losses.append(0)

        return self._trajectory


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
    free_x, free_y = np.where(board[2, :, :].numpy() == 1)
    return [coord_to_action(board, [x, y]) for x, y in zip(free_x, free_y)]


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

    def evaluate(self, example):
        return -example.reward


class HexFeatures(macarico.DynamicFeatures):
    def __init__(self, board_size=5):
        self.board_size = board_size
        macarico.DynamicFeatures.__init__(self, 3 * self.board_size ** 2)

    def _forward(self, state):
        view = state.state.view(1, 1, 3 * self.board_size ** 2)
        return Varng(view)
