from __future__ import division

import random
import sys
import torch
from torch.autograd import Variable
import macarico
import torch.nn.functional as F

class BanditLOLS(macarico.Learner):
    MIX_PER_STATE, MIX_PER_ROLL = 0, 1
    LEARN_BIASED, LEARN_IPS, LEARN_DR, LEARN_MTR, LEARN_MTR_ADVANTAGE, _LEARN_MAX = 0, 1, 2, 3, 4, 5
    EXPLORE_UNIFORM, EXPLORE_BOLTZMANN, EXPLORE_BOLTZMANN_BIASED, _EXPLORE_MAX = 0, 1, 2, 3

    def __init__(self, reference, policy, p_rollin_ref, p_rollout_ref,
                 learning_method=LEARN_IPS,
                 exploration=EXPLORE_UNIFORM, baseline=None,
                 epsilon=1.0, mixture=MIX_PER_ROLL, use_prefix_costs=False,
                 temperature=1., loglinear_policy=False):
        self.reference = reference
        self.policy = policy
        self.learning_method = learning_method
        self.exploration = exploration
        self.temperature = temperature
        self.loglinear_policy = loglinear_policy
        assert self.learning_method in range(BanditLOLS._LEARN_MAX), \
            'unknown learning_method, must be one of BanditLOLS.LEARN_*'
        assert self.exploration in range(BanditLOLS._EXPLORE_MAX), \
            'unknown exploration, must be one of BanditLOLS.EXPLORE_*'
        
        self.use_prefix_costs = use_prefix_costs
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
        self.dev_actions = None
        self.dev_imp_weight = None
        self.dev_costs = None
        self.squared_loss = 0.
        self.pred_cost_without_dev = 0.
        
        super(BanditLOLS, self).__init__()

    def __call__(self, state):
        if self.t is None:
            self.t = 0
            self.dev_t = random.randint(1, state.T)

        self.t += 1
        a_ref = self.reference(state)
        if self.t == self.dev_t:
            if random.random() > self.epsilon: # exploit
                return self.policy(state)
            else:
                self.dev_costs = self.policy.predict_costs(state)
                self.dev_actions = list(state.actions)[:]
                self.dev_a, self.dev_imp_weight = self.explore(self.dev_costs)
                return self.dev_a if isinstance(self.dev_a, int) else self.dev_a.data[0,0]
        else:
            pred_costs = self.policy.predict_costs(state)
            a = a_ref
            if not (self.rollin_ref() if self.t < self.dev_t else self.rollout_ref()):
                a = self.policy.greedy(state, pred_costs)
            self.pred_cost_without_dev += pred_costs.data[0, a]
            return a

    def explore(self, costs):
        # returns action and importance weight
        if self.exploration == BanditLOLS.EXPLORE_UNIFORM:
            return random.choice(list(self.dev_actions)), len(self.dev_actions)
        if self.exploration in [BanditLOLS.EXPLORE_BOLTZMANN, BanditLOLS.EXPLORE_BOLTZMANN_BIASED]:
            if len(self.dev_actions) != len(costs):
                for i in xrange(len(costs)):
                    if i not in self.dev_actions:
                        costs[0,i] = 1e10
            #costs = -costs / self.temperature
            #shift = costs.max()
            #costs -= shift
            #costs = costs.exp()
            #costs /= costs.sum()
            #a = costs.multinomial(1)
            a = F.softmax(- costs / self.temperature).multinomial()
            p = F.softmax(- costs / self.temperature).data[0, a.data[0,0]]
            if self.exploration == BanditLOLS.EXPLORE_BOLTZMANN_BIASED:
                p = max(p, 1e-4)
            return a, 1 / p
        assert False, 'unknown exploration strategy'
    
        
    def update(self, loss):
        if self.use_prefix_costs:
            loss -= self.pred_cost_without_dev
            
        if self.dev_a is not None:
            baseline = 0 if self.baseline is None else self.baseline()
            truth = self.build_cost_vector(baseline, loss)
            importance_weight = 1
            old_dev_actions = self.dev_actions[:]
            if self.learning_method in [BanditLOLS.LEARN_MTR, BanditLOLS.LEARN_MTR_ADVANTAGE]:
                self.dev_actions = [self.dev_a if isinstance(self.dev_a, int) else self.dev_a.data[0,0]]
                importance_weight = self.dev_imp_weight
            #print 'diff = %s, a = %s' % (self.dev_costs.data - truth, self.dev_actions)
            #print (self.dev_costs.data - truth)[0,self.dev_actions[0]]
            if not self.loglinear_policy:
                loss_var = self.policy.forward_partial_complete(self.dev_costs, truth, self.dev_actions)
                self.dev_actions = old_dev_actions # TODO remove?
                loss_var *= importance_weight
                loss_var.backward()
            else: # loglinear_policy
                self.dev_a.reinforce(loss - baseline)
            a = self.dev_a if isinstance(self.dev_a, int) else self.dev_a.data[0,0]
            self.squared_loss = (loss - self.dev_costs.data[0, a]) ** 2
            
            if self.baseline is not None:
                self.baseline.update(loss)

    def build_cost_vector(self, baseline, loss):
        costs = torch.zeros(self.policy.n_actions)
        a = self.dev_a
        if not isinstance(a, int):
            a = a.data[0,0]
        if self.learning_method == BanditLOLS.LEARN_BIASED:
            costs -= baseline
            costs[a] = loss - baseline
        elif self.learning_method == BanditLOLS.LEARN_IPS:
            costs -= baseline
            costs[a] = (loss - baseline) * self.dev_imp_weight
        elif self.learning_method == BanditLOLS.LEARN_DR:
            costs += self.dev_costs.data # now costs = \hat c
            costs[a] = self.dev_costs.data[0,a] + self.dev_imp_weight * (loss - self.dev_costs.data[0,a])
        elif self.learning_method == BanditLOLS.LEARN_MTR:
            costs[a] = loss - baseline
        elif self.learning_method == BanditLOLS.LEARN_MTR_ADVANTAGE:
            costs[a] = loss - self.dev_costs.data.min()
        else:
            assert False, self.learning_method
        return costs

class EpisodeRunner(macarico.Learner):
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
        self.ref_costs = []

    def __call__(self, state):
        a_type = self.run_strategy(self.t)
        pol = self.policy(state)
        ref_costs_t = torch.zeros(self.policy.n_actions)
        self.reference.set_min_costs_to_go(state, ref_costs_t)
        self.ref_costs.append(ref_costs_t)
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
        cost = self.policy.predict_costs(state)
        self.costs.append( cost )
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
    
def lols(ex, loss, ref, policy, p_rollin_ref, p_rollout_ref,
         mixture=BanditLOLS.MIX_PER_ROLL):
    # construct the environment
    env = ex.mk_env()
    # set up a helper function to run a single trajectory
    def run(run_strategy):
        env.rewind()
        runner = EpisodeRunner(policy, run_strategy, ref)
        env.run_episode(runner)
        cost = loss()(ex, env)
        return cost, runner.trajectory, runner.limited_actions, runner.costs

    n_actions = env.n_actions
    
    # construct rollin and rollout policies
    if mixture == BanditLOLS.MIX_PER_STATE:
        # initialize tied randomness for both rollin and rollout
        # TODO THIS IS BORKEN!
        rng = TiedRandomness()
        rollin_f  = lambda t: EpisodeRunner.REF if rng(t) <= p_rollin_ref  else EpisodeRunner.LEARN
        rollout_f = lambda t: EpisodeRunner.REF if rng(t) <= p_rollout_ref else EpisodeRunner.LEARN
    else:
        rollin  = EpisodeRunner.REF if p_rollin_ref()  else EpisodeRunner.LEARN
        rollout = EpisodeRunner.REF if p_rollout_ref() else EpisodeRunner.LEARN
        rollin_f  = lambda t: rollin
        rollout_f = lambda t: rollout

    # build a back-bone using rollin policy
    loss0, traj0, limit0, costs0 = run(rollin_f)

    # start one-step deviations
    objective = 0. # Variable(torch.zeros(1))
    traj_rollin = lambda t: (EpisodeRunner.ACT, traj0[t])
    for t, costs_t in enumerate(costs0):
        costs = torch.zeros(n_actions)
        # collect costs for all possible actions
        for a in limit0[t]:
            l, _, _, _ = run(one_step_deviation(traj_rollin, rollout_f, t, a))
            costs[a] = l
        # accumulate update
        costs -= min(costs)
        objective += policy.forward_partial_complete(costs_t, costs, limit0[t])

    # run backprop
    objective.backward()

    return objective

