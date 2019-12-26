import random

import torch.nn as nn

import macarico
import macarico.util as util


class GridSettings(macarico.Example):
    def __init__(self, width, height, start, walls, terminals, per_step_cost, max_steps, gamma, p_step_success,
                 do_break=True):
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
        self.do_break = do_break


def make_default_gridworld(per_step_cost=0.05, max_steps=50, gamma=0.99, p_step_success=0.8, start_random=False):
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


def make_debug_gridworld(per_step_cost=0.05, max_steps=6, gamma=1.0, p_step_success=1.0, start_random=False):
    #    0123
    #   0   +
    #   1 # -
    #   2 #
    #   3
    start = (0, 3)
    if start_random:
        start = (random.randint(0, 3), random.randint(0, 3))
    do_break = True
    return GridWorld(GridSettings(4, 4, start, {(1, 1), (1, 2)}, {(3, 0): 1, (3, 1): -1},
                                  per_step_cost, max_steps, gamma, p_step_success, do_break))


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
                if self.example.do_break:
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
        return ((new_loc[0] >= 0) and (new_loc[0] < self.example.width)) and \
               ((new_loc[1] >= 0) and (new_loc[1] < self.example.height)) and \
               (new_loc not in self.example.walls)

    def get_x_y(self, state_id):
        y = int(state_id % self.example.height)
        x = int((state_id - y) / self.example.height)
        return x, y

    def get_state_id(self, x, y):
        return x * self.example.height + y

    def x_y_step(self, x, y, a):
        new_loc = [x, y]
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
            return new_loc
        else:
            return (x, y)

    def make_step(self, state, action):
        x, y = self.get_x_y(state)
        new_x, new_y = self.x_y_step(x, y, action)
        return self.get_state_id(new_x, new_y)

    def costs_function(self):
        import numpy as np
        costs = np.zeros((16, 4))
        for current_state in range(16):
            for action in range(4):
                next_state = self.make_step(current_state, action)
                loc = self.get_x_y(next_state)
                costs[current_state, action] = costs[current_state, action] + self.example.per_step_cost
                if loc in self.example.terminal:
                    costs[current_state, action] -= self.example.terminal[loc]
        return costs

    def costs(self, pi):
        import numpy as np
        costs = np.zeros(16)
        for current_state, action_distribution in enumerate(pi):
            for action, action_probability in enumerate(action_distribution):
                next_state = self.make_step(current_state, action)
                loc = self.get_x_y(next_state)
                costs[current_state] = costs[current_state] + action_probability * self.example.per_step_cost
                if loc in self.example.terminal:
                    costs[current_state] -= action_probability * self.example.terminal[loc]
        return costs

    def transition(self):
        import numpy as np
        model = np.zeros((16, 4, 16))
        for current_state in range(16):
            for action in range(4):
                next_state = self.make_step(current_state, action)
                model[current_state, action, next_state] = 1.0
        return model

    def model(self, pi):
        import numpy as np
        model = np.zeros((16, 16))
        for current_state, action_distribution in enumerate(pi):
            for action, action_probability in enumerate(action_distribution):
                next_state = self.make_step(current_state, action)
                model[current_state, next_state] += action_probability
        return model

    def policy_eval(self, policy, P, rewards, discount_factor=1.0, theta=0.00001):
        """
        Evaluate a policy given an environment and a full description of the environment's dynamics.

        Args:
            policy: [S, A] shaped matrix representing the policy.
            env: OpenAI env. env.P represents the transition probabilities of the environment.
                env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
                env.nS is a number of states in the environment.
                env.nA is a number of actions in the environment.
            theta: We stop evaluation once our value function change is less than theta for all states.
            discount_factor: Gamma discount factor.

        Returns:
            Vector of length env.nS representing the value function.
        """
        import numpy as np
        # Start with a random (all 0) value function
        # TODO Generalize to different number of states
        nS = 16
        V = np.zeros(nS)
        for _ in range(self.example.max_steps):
            V_new = np.zeros(nS)
            delta = 0
            # For each state, perform a "full backup"
            for s in range(nS):
                v = 0
                # Look at the possible next actions
                for a, action_prob in enumerate(policy[s]):
                    # For each action, look at the possible next states...
                    for next_state, prob in enumerate(P[s, a]):
                        reward = rewards[s, a]
                        # Calculate the expected value. Ref: Sutton book eq. 4.6.
                        v += action_prob * prob * (reward + discount_factor * V[next_state])
                # How much our value function changed (across any states)
                delta = max(delta, np.abs(v - V[s]))
                V_new[s] = v
            V = V_new
            # Stop evaluating once our value function change is below a threshold
            if delta < theta:
                break
        return np.array(V)

    def fin_horizon_VI(self, policy, P, rewards, horizon, discount_factor=1.0):
        import numpy as np
        V = []
        Q = []
        nS = 16
        V.append(np.zeros(nS))
        Q.append(P.dot(np.zeros(nS)))
        for steps in range(horizon):
            V_new = np.zeros(nS)
            V_prev = V[-1]
            for s in range(nS):
                v = 0
                for a, action_prob in enumerate(policy[s]):
                    for next_state, prob in enumerate(P[s, a]):
                        reward = rewards[s, a]
                        v += action_prob * prob * (reward + discount_factor * V_prev[next_state])
                V_new[s] = v
            V.append(V_new)
            Q.append(rewards + discount_factor * P.dot(V_prev))
        return V, Q



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
        return view

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
        return view

    def __call__(self, state):
        return self.forward(state)
