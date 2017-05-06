from __future__ import division

import random
import sys
import torch
from torch.autograd import Variable
import macarico

def ZeroBaseline():
    return 0.0

class BanditLOLS(macarico.LearningAlg):
    MIX_PER_STATE, MIX_PER_ROLL = 0, 1
    LEARN_REINFORCE, LEARN_IMPORTANCE = 0, 1

    def __init__(self, reference, policy, p_rollin_ref, p_rollout_ref,
                 learning_method=LEARN_REINFORCE, baseline=ZeroBaseline,
                 epsilon=1.0, mixture=MIX_PER_ROLL, save_costs=False):
        self.reference = reference
        self.policy = policy
        self.learning_method = learning_method
        self.save_costs = save_costs
        self.costs = []
        if mixture == BanditLOLS.MIX_PER_ROLL:
            use_in_ref  = p_rollin_ref()
            use_out_ref = p_rollout_ref()
            self.rollin_ref  = lambda: use_in_ref
            self.rollout_ref = lambda: use_out_ref
        else:
            self.rollin_ref  = p_rollin_ref
            self.rollout_ref = p_rollout_ref
        self.baseline = baseline
        self.epsilon = epsilon
        self.t = None
        self.dev_t = None
        self.dev_a = None
        self.dev_weight = None
        self.dev_state = None
        self.dev_limit_actions = None
        
        super(BanditLOLS, self).__init__()

    def __call__(self, state):
        if self.t is None:
            self.t = 0
            self.dev_t = random.randint(1, state.T)

        self.t += 1        
        if self.t == self.dev_t:
            if random.random() > self.epsilon: # exploit
                return self.policy(state)
            elif self.learning_method == BanditLOLS.LEARN_REINFORCE:
                self.dev_state = self.policy.stochastic(state)
                self.dev_a = self.dev_state.data[0,0]
                return self.dev_a
            elif self.learning_method == BanditLOLS.LEARN_IMPORTANCE:
                self.dev_a = random.choice(state.actions)
                self.dev_weight = len(state.actions)
                self.dev_state = self.policy.predict_costs(state)
                self.dev_limit_actions = list(state.actions)
                return self.dev_a
        elif self.rollin_ref() if self.t < self.dev_t else self.rollout_ref():
            self.policy(state) # must call this to get updates
            return self.reference(state)
        else:
            return self.policy(state)

    def update(self, loss):
        if self.dev_a is not None:
            if self.learning_method == BanditLOLS.LEARN_REINFORCE:
                self.dev_state.reinforce(self.baseline() - loss)
                torch.autograd.backward(self.dev_state, [None])
            elif self.learning_method == BanditLOLS.LEARN_IMPORTANCE:
                truth = self.build_cost_vector(self.baseline(), loss)
                self.policy.forward_partial_complete(self.dev_state, truth).backward()
        if self.baseline is not None:
            self.baseline.update(loss)

    def build_cost_vector(self, baseline, loss):
        # TODO: handle self.dev_limit_actions
        # TODO: doubly robust
        costs = torch.zeros(self.policy.n_actions) + baseline
        costs[self.dev_a] = loss * self.dev_weight
        return costs

class EpisodeRunner(macarico.LearningAlg):
    REF, LEARN, ACT = 0, 1, 2

    def __init__(self, policy, run_strategy, reference=None):
        self.policy = policy
        self.run_strategy = run_strategy
        self.reference = reference
        self.t = 0
        self.total_loss = 0.
        self.trajectory = []
        self.limited_actions = []
        self.costs = []

    def __call__(self, state):
        a_type = self.run_strategy(self.t)
        pol = self.policy(state)
        if a_type == EpisodeRunner.REF:
            a = self.reference(state)
        elif a_type == EpisodeRunner.LEARN:
            a = pol
        elif isinstance(a_type, tuple) and a_type[0] == EpisodeRunner.ACT:
            a = a_type[1]
        else:
            raise ValueError('run_strategy yielded an invalid choice %s' % a_type)

        assert a in state.actions, \
            'EpisodeRunner strategy insisting on an illegal action :('

        self.limited_actions.append(state.actions)
        self.trajectory.append(a)
        self.costs.append( self.policy.predict_costs(state) )
        self.t += 1

        return a
    
def one_step_deviation(rollin, rollout, dev_t, dev_a):
    return lambda t: \
        (EpisodeRunner.ACT, dev_a) if t == dev_t else \
        rollin(t) if t < dev_t else \
        rollout(t)

class TiedRandomness(object):
    def __init__(self, rng=random.random):
        self.tied = {}
        self.rng = rng

    def reset(self):
        self.tied = {}

    def __call__(self, t):
        if t not in self.tied:
            self.tied[t] = self.rng()
        return self.tied[t]

def lols(ex, policy, p_rollin_ref, p_rollout_ref,
         mixture=BanditLOLS.MIX_PER_ROLL):
    # construct the environment
    env = ex.mk_env()
    # set up a helper function to run a single trajectory
    def run(run_strategy):
        env.rewind()
        runner = EpisodeRunner(policy, run_strategy, env.reference())
        env.run_episode(runner)
        cost = env.loss()
        return cost, runner.trajectory, runner.limited_actions, runner.costs

    n_actions = env.n_actions
    
    # construct rollin and rollout policies
    if mixture == BanditLOLS.MIX_PER_STATE:
        # initialize tied randomness for both rollin and rollout
        rng = TiedRandomness()
        rollin_f  = lambda t: EpisodeRunner.REF if rng(t) <= p_rollin_ref  else EpisodeRunner.LEARN
        rollout_f = lambda t: EpisodeRunner.REF if rng(t) <= p_rollout_ref else EpisodeRunner.LEARN
    else:
        rollin  = EpisodeRunner.REF if random.random() <= p_rollin_ref  else EpisodeRunner.LEARN
        rollout = EpisodeRunner.REF if random.random() <= p_rollout_ref else EpisodeRunner.LEARN
        rollin_f  = lambda t: rollin
        rollout_f = lambda t: rollout

    # build a back-bone using rollin policy
    loss0, traj0, limit0, costs0 = run(rollin_f)

    # start one-step deviations
    objective = 0. # Variable(torch.zeros(1))
    traj_rollin = lambda t: (EpisodeRunner.ACT, traj0[t])
    for t, costs_t in enumerate(costs0):
        costs = torch.zeros(1,n_actions)
        # collect costs for all possible actions
        for a in limit0[t]:
            l, _, _, _ = run(one_step_deviation(traj_rollin, rollout_f, t, a))
            costs[0,a] = l
        # accumulate update
        costs -= min(costs[0])
        objective += policy.forward_partial_complete(costs_t, costs, limit0[t])

    # run backprop
    objective.backward()
